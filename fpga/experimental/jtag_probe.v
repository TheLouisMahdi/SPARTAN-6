// Isolated USER1 transport probe. This module does not contain or claim a CNN.
// Request: 16 zero prefix bytes, <version:u8,command:u8,seq:u32,length:u16>,
// payload, CCITT-FALSE CRC16 little endian, one zero tail byte. Bits are LSB first.
// Reply: <"JTG1",seq:u32,length:u16,crc:u16,status:u8,version:u8,accepted:u16>.
// TDO first emits 16 reply bytes, then a one-bit delayed TDI echo.
module jtag_probe(input clk,output wire [3:0] led);
    wire capture,drck,tap_reset,runtest,sel,shift,tck_raw,tdi,tms,update,tdo,tck;
    BSCAN_SPARTAN6 #(.JTAG_CHAIN(1)) user1(.CAPTURE(capture),.DRCK(drck),
        .RESET(tap_reset),.RUNTEST(runtest),.SEL(sel),.SHIFT(shift),.TCK(tck_raw),
        .TDI(tdi),.TMS(tms),.UPDATE(update),.TDO(tdo));
    BUFG scan_clock(.I(tck_raw),.O(tck));
    jtag_probe_logic stream(clk,tck,tap_reset,sel,capture,shift,update,tdi,tdo,led);
endmodule

module jtag_probe_logic(input clk,input tck,input tap_reset,input sel,
    input capture,input shift,input update,input tdi,output wire tdo,output wire [3:0] led);
    `include "rtl/crc16_function.vh"
    reg [1:0] trst=3,srst=3;
    always @(posedge tck or posedge tap_reset)if(tap_reset)trst<=3;else trst<={trst[0],1'b0};
    always @(posedge clk or posedge tap_reset)if(tap_reset)srst<=3;else srst<={srst[0],1'b0};
    wire wrst=trst[1],rrst=srst[1];
    reg [2:0] bit_index=0;
    reg [7:0] rx_shift=0;
    wire [7:0] incoming={tdi,rx_shift[7:1]};
    wire fifo_write=sel&&(capture||update||(shift&&bit_index==7));
    wire [8:0] fifo_in=capture ? 9'h100 : update ? (bit_index==0 ? 9'h101 : 9'h102) : {1'b0,incoming};
    wire [8:0] fifo_out;
    wire full,empty,overflow;
    jtag_async_fifo fifo(tck,wrst,fifo_write,fifo_in,full,overflow,clk,rrst,!empty,fifo_out,empty);
    reg [12:0] scan_bytes=0;
    reg [7:0] out_bits=0;
    reg echo_bit=0;
    reg [127:0] tx_shift=0;
    reg [127:0] reply=128'h0000010900000000000000003147544a;
    reg ack=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg ack_meta=0,ack_sync=0;
    reg ack_seen=0,reply_valid=1;
    assign tdo=(out_bits<128) ? tx_shift[0] : echo_bit;
    always @(posedge tck or posedge wrst)begin
        if(wrst)begin bit_index<=0;rx_shift<=0;scan_bytes<=0;out_bits<=0;echo_bit<=0;
            tx_shift<=0;ack_meta<=0;ack_sync<=0;ack_seen<=0;reply_valid<=1;end
        else begin
            ack_meta<=ack;ack_sync<=ack_meta;
            if(ack_sync!=ack_seen)begin ack_seen<=ack_sync;reply_valid<=1;end
            if(sel&&capture)begin
                bit_index<=0;scan_bytes<=0;out_bits<=0;echo_bit<=0;
                // Bundled-data mailbox: one request credit. reply is held until
                // the next nonzero request; ACK has crossed two TCK registers.
                tx_shift<=reply_valid ? reply : 128'h0000010800000000000000003147544a;
            end else if(sel&&shift)begin
                echo_bit<=tdi;rx_shift<=incoming;bit_index<=bit_index+1'b1;
                if(out_bits<128)begin tx_shift<={1'b0,tx_shift[127:1]};out_bits<=out_bits+1'b1;end
                if(bit_index==7)begin
                    if(scan_bytes==16&&incoming!=0)reply_valid<=0;
                    if(scan_bytes!=8191)scan_bytes<=scan_bytes+1'b1;
                end
            end
        end
    end
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg overflow_meta=0,overflow_sync=0;
    reg [12:0] count=0;
    reg [15:0] length=0,crc=16'hffff,received_crc=0,accepted=0;
    reg [31:0] sequence_id=0,last_sequence=0;
    reg seen=0,is_request=0,crc_ok=0;
    reg [7:0] command=0,error_code=0;
    wire [7:0] data=fifo_out[7:0];
    wire [7:0] terminal_status=overflow_sync ? 8'd4 : fifo_out==9'h102 ? 8'd5 :
        count!=length+27 || (command==3&&length!=0) ? 8'd2 : error_code!=0 ? error_code : !crc_ok ? 8'd1 :
        command!=1&&command!=3 ? 8'd7 :
        command!=3&&seen&&sequence_id!=last_sequence+1'b1 ? 8'd3 : 8'd0;
    assign led={overflow_sync,seen,~empty,accepted[0]};
    always @(posedge clk or posedge rrst)begin
        if(rrst)begin count<=0;length<=0;crc<=16'hffff;received_crc<=0;accepted<=0;
            sequence_id<=0;last_sequence<=0;seen<=0;is_request<=0;crc_ok<=0;
            command<=0;error_code<=0;overflow_meta<=0;overflow_sync<=0;ack<=0;
            reply<=128'h0000010900000000000000003147544a;end
        else begin
            overflow_meta<=overflow;overflow_sync<=overflow_meta;
            if(!empty)begin
                if(fifo_out==9'h100)begin
                    count<=0;length<=0;crc<=16'hffff;received_crc<=0;
                    sequence_id<=0;is_request<=0;crc_ok<=0;command<=0;error_code<=0;
                end else if(fifo_out[8])begin
                    if(is_request||fifo_out==9'h102)begin
                        reply<={accepted+(terminal_status==0),8'd1,terminal_status,crc,length,sequence_id,32'h3147544a};
                        ack<=~ack;
                        if(terminal_status==0)begin accepted<=accepted+1'b1;seen<=1;last_sequence<=sequence_id;end
                    end
                end else begin
                    if(count!=8191)count<=count+1'b1;else error_code<=2;
                    if(count>=16&&count<24)crc<=crc16_byte(crc,data);
                    if(count==16)begin is_request<=(data!=0);if(data!=0&&data!=1)error_code<=6;end
                    else if(count==17)command<=data;
                    else if(count>=18&&count<22)sequence_id[(count-18)*8 +: 8]<=data;
                    else if(count==22)length[7:0]<=data;
                    else if(count==23)begin
                        length[15:8]<=data;if({data,length[7:0]}>6144)error_code<=2;
                    end else if(count>=24&&count<24+length)crc<=crc16_byte(crc,data);
                    else if(count==24+length)received_crc[7:0]<=data;
                    else if(count==25+length)begin received_crc[15:8]<=data;crc_ok<=({data,received_crc[7:0]}==crc);end
                    else if(count==26+length&&data!=0)error_code<=2;
                end
            end
        end
    end
endmodule
