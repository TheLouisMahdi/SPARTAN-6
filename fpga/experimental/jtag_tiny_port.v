// Experimental full-CNN transport candidate. Never a replacement for baseline
// until routing, protocol simulation and physical gates pass.
module jtag_tiny_port(input clk,input available,input model_loaded,input [7:0] active_method,input [31:0] active_fingerprint,input cnn_busy,input cnn_done,
    input signed [31:0] cnn_first,input signed [31:0] cnn_second,input [31:0] cnn_cycles,
    output reg claim=0,output reg input_we=0,output reg [11:0] input_address=0,
    output reg [7:0] input_data=0,output reg start=0);
    `include "rtl/crc16_function.vh"
    wire capture,drck,tap_reset,runtest,sel,shift,tck_raw,tdi,tms,update,tdo,tck;
    BSCAN_SPARTAN6 #(.JTAG_CHAIN(1)) user1(.CAPTURE(capture),.DRCK(drck),.RESET(tap_reset),
        .RUNTEST(runtest),.SEL(sel),.SHIFT(shift),.TCK(tck_raw),.TDI(tdi),.TMS(tms),.UPDATE(update),.TDO(tdo));
    BUFG scan_clock(.I(tck_raw),.O(tck));
    reg [1:0] trst=3,srst=3;
    always @(posedge tck or posedge tap_reset)if(tap_reset)trst<=3;else trst<={trst[0],1'b0};
    always @(posedge clk or posedge tap_reset)if(tap_reset)srst<=3;else srst<={srst[0],1'b0};
    wire wrst=trst[1],rrst=srst[1];
    reg [2:0] bit_index=0;
    reg [7:0] rx_shift=0;
    wire [7:0] incoming={tdi,rx_shift[7:1]};
    wire fifo_write=sel&&(capture||update||(shift&&bit_index==7));
    wire [8:0] fifo_in=capture ? 9'h100 : update ? (bit_index==0 ? 9'h101 : 9'h102) : {1'b0,incoming};
    wire [8:0] fifo_out;wire full,empty,overflow;
    reg [2:0] expand_left=0;reg [7:0] symbols=0,encoder_id=0;reg [8:0] expand_address=0;
    jtag_async_fifo fifo(tck,wrst,fifo_write,fifo_in,full,overflow,clk,rrst,!empty&&expand_left==0,fifo_out,empty);
    reg [12:0] scan_bytes=0;
    reg [8:0] out_bits=0;
    reg echo_bit=0;
    reg [327:0] tx_shift=0;
    reg [327:0] reply=328'h0000020900000000000000003159544a;
    reg ack=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg ack_meta=0,ack_sync=0;
    reg ack_seen=0,reply_valid=1;
    assign tdo=(out_bits<328) ? tx_shift[0] : echo_bit;
    always @(posedge tck or posedge wrst)begin
        if(wrst)begin bit_index<=0;rx_shift<=0;scan_bytes<=0;out_bits<=0;echo_bit<=0;
            tx_shift<=0;ack_meta<=0;ack_sync<=0;ack_seen<=0;reply_valid<=1;end
        else begin
            ack_meta<=ack;ack_sync<=ack_meta;
            if(ack_sync!=ack_seen)begin ack_seen<=ack_sync;reply_valid<=1;end
            if(sel&&capture)begin
                bit_index<=0;scan_bytes<=0;out_bits<=0;echo_bit<=0;
                tx_shift<=reply_valid ? reply : 328'h0000020800000000000000003159544a;
            end else if(sel&&shift)begin
                echo_bit<=tdi;rx_shift<=incoming;bit_index<=bit_index+1'b1;
                if(out_bits<328)begin tx_shift<={1'b0,tx_shift[327:1]};out_bits<=out_bits+1'b1;end
                if(bit_index==7)begin
                    if(scan_bytes==41&&incoming!=0)reply_valid<=0;
                    if(scan_bytes!=8191)scan_bytes<=scan_bytes+1'b1;
                end
            end
        end
    end
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg overflow_meta=0,overflow_sync=0;
    // The largest legal scan is 198 bytes. Saturate oversized scans instead
    // of synthesizing the reference transport's 8 KiB address arithmetic.
    reg [7:0] count=0;
    reg [15:0] length=0,crc=16'hffff,crc_low=0,accepted=0,pending_crc=0;
    reg [31:0] sequence_id=0,last_sequence=0,pending_sequence=0;
    reg [31:0] frame_id=0,model_id=0;
    reg [63:0] capture_ns=0;
    reg [7:0] eye_id=0;
    reg seen=0,is_request=0,crc_ok=0,inflight=0,granted=0;
    reg [7:0] command=0,error_code=0;
    wire [7:0] data=fifo_out[7:0];
    wire [7:0] terminal_status=overflow_sync ? 8'd4 : fifo_out==9'h102 ? 8'd5 :
        count!=length+52 || (command==3&&length!=0) || (command==2&&length!=146) ? 8'd2 :
        error_code!=0 ? error_code : !crc_ok ? 8'd1 :
        command!=2&&command!=3 ? 8'd7 : inflight ? 8'd8 :
        command==2&&(!granted||cnn_busy) ? 8'd8 :
        command==2&&(!model_loaded||active_method!=1||model_id!=active_fingerprint) ? 8'd9 :
        command==2&&encoder_id!=1 ? 8'd2 :
        command!=3&&seen&&sequence_id!=last_sequence+1'b1 ? 8'd3 : 8'd0;
    always @(posedge clk or posedge rrst)begin
        if(rrst)begin count<=0;length<=0;crc<=16'hffff;crc_low<=0;accepted<=0;
            sequence_id<=0;last_sequence<=0;seen<=0;is_request<=0;crc_ok<=0;
            command<=0;error_code<=0;expand_left<=0;symbols<=0;expand_address<=0;encoder_id<=0;overflow_meta<=0;overflow_sync<=0;ack<=0;
            reply<=328'h0000020900000000000000003159544a;claim<=0;input_we<=0;
            input_address<=0;input_data<=0;start<=0;inflight<=0;granted<=0;
            pending_sequence<=0;pending_crc<=0;frame_id<=0;capture_ns<=0;eye_id<=0;model_id<=0;
        end else begin
            input_we<=0;start<=0;overflow_meta<=overflow;overflow_sync<=overflow_meta;
            if(expand_left!=0)begin
                input_we<=1;input_address<={3'd0,expand_address};
                case(symbols[7:6])0:input_data<=0;1:input_data<=85;2:input_data<=170;3:input_data<=255;endcase
                symbols<={symbols[5:0],2'b0};expand_left<=expand_left-1'b1;expand_address<=expand_address+1'b1;
            end
            if(!empty&&expand_left==0)begin
                if(fifo_out==9'h100)begin
                    count<=0;length<=0;crc<=16'hffff;crc_low<=0;sequence_id<=0;
                    is_request<=0;crc_ok<=0;command<=0;error_code<=0;granted<=0;
                end else if(fifo_out[8])begin
                    if(is_request||fifo_out==9'h102)begin
                        if(terminal_status==0&&command==2)begin
                            start<=1;inflight<=1;pending_sequence<=sequence_id;pending_crc<=crc;
                        end else begin
                            reply<={200'd0,accepted+(terminal_status==0),8'd2,terminal_status,crc,length,sequence_id,32'h3159544a};
                            ack<=~ack;if(!inflight)claim<=0;
                        end
                        if(terminal_status==0)begin accepted<=accepted+1'b1;seen<=1;last_sequence<=sequence_id;end
                    end
                end else begin
                    if(count!=255)count<=count+1'b1;else error_code<=2;
                    if(count>=41&&count<49)crc<=crc16_byte(crc,data);
                    if(count==41)begin is_request<=(data!=0);if(data!=0&&data!=2)error_code<=6;end
                    else if(count==42)begin
                        command<=data;
                        if(data==2)begin
                            if(available&&!inflight)begin claim<=1;granted<=1;end
                            else error_code<=8;
                        end
                    end else if(count>=43&&count<47)sequence_id[(count-43)*8 +: 8]<=data;
                    else if(count==47)length[7:0]<=data;
                    else if(count==48)begin length[15:8]<=data;if({data,length[7:0]}>146)error_code<=2;end
                    else if(count>=49&&count<49+length)begin
                        crc<=crc16_byte(crc,data);
                        if(granted&&!inflight&&command==2)begin
                            if(count<53)frame_id[(count-49)*8 +: 8]<=data;
                            else if(count<61)capture_ns[(count-53)*8 +: 8]<=data;
                            else if(count==61)begin eye_id<=data;if(data>1)error_code<=2;end
                            else if(count<66)model_id[(count-62)*8 +: 8]<=data;
                            else if(count==66)begin encoder_id<=data;if(data!=1)error_code<=2;end
                            else if(count<195&&!cnn_busy&&model_loaded&&active_method==1&&model_id==active_fingerprint&&encoder_id==1&&error_code==0)begin
                                symbols<=data;expand_address<=(count-67)*4;expand_left<=4;
                            end
                        end
                    end else if(count==49+length)crc_low[7:0]<=data;
                    else if(count==50+length)crc_ok<=({data,crc_low[7:0]}==crc);
                    else if(count==51+length&&data!=0)error_code<=2;
                end
            end
            if(inflight&&cnn_done)begin
                // Metadata and native logits remain associated with one eye;
                // TinyEye native order is OPEN,CLOSED; no host reversal.
                reply<={eye_id,capture_ns,frame_id,cnn_cycles,cnn_second,cnn_first,
                    accepted,8'd2,8'd0,pending_crc,16'd146,pending_sequence,32'h3159544a};
                ack<=~ack;inflight<=0;claim<=0;
            end
        end
    end
endmodule
