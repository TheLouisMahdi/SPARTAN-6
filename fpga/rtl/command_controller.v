// Packet command sequencing, payload access and cycle-preserving CNN handshakes.
module command_controller #(parameter integer PRETRAINED=1)(
    input wire clk, input wire reset,
    input wire packet_valid, input wire [7:0] packet_kind,
    input wire [31:0] packet_sequence, input wire [15:0] packet_length,
    output reg rx_accept=0, output reg [10:0] read_address=0, input wire [7:0] read_data,
    output reg tx_start=0, input wire tx_packet_done,
    output wire [7:0] reply_kind, output wire [31:0] reply_sequence,
    output reg [5:0] reply_length=1, input wire [5:0] reply_address, output wire [7:0] reply_data,
    output reg model_we=0, output reg [10:0] model_address=0, output reg [7:0] model_data=0,
    output reg input_we=0, output reg [11:0] input_address=0, output reg [7:0] input_data=0,
    output reg core_start=0, input wire core_done,
    input wire signed [31:0] core_open, input wire signed [31:0] core_closed,
    input wire [31:0] core_cycles, output reg [10:0] debug_address=0, input wire [7:0] debug_data,
    input wire [3:0] key_pulse, output reg [1:0] page=0,
    output reg result_valid=0, output reg [3:0] result_flags=0,
    output reg [23:0] frames=0, output reg [23:0] last_pair_cycles=0,
    output wire [3:0] decisions, output reg timing_mode=0,
    output reg [31:0] capture_ms=0, output reg [1:0] timing_valid=0,
    output reg [15:0] left_score=0, output reg [15:0] right_score=0,
    output wire clear_blinks, output wire invalidate_blinks, output wire start_pair,
    output wire pair_active, output wire result_update, output wire timed_result,
    output wire timing_update, output wire timing_reply,
    input wire [1:0] blink_events, input wire [31:0] left_blinks, input wire [31:0] right_blinks,
    input wire [1:0] timing_closed, input wire [1:0] timing_known, input wire timing_event,
    input wire [31:0] timing_count, input wire [31:0] timing_long,
    input wire [31:0] left_current_ms, input wire [31:0] right_current_ms,
    input wire [31:0] last_left_ms, input wire [31:0] last_right_ms,
    input wire [31:0] last_overlap_ms, input wire [31:0] last_duration_ms);
    `include "generated/model_config.vh"
    `include "generated/omz_config.vh"
    localparam IDLE=0,DISPATCH=1,READ_REQ=2,READ_WAIT=3,READ_CAP=4,
        MODEL_COMMIT=5,CONFIG_COMMIT=6,EYE_REQ=7,EYE_WAIT_READ=8,EYE_CAPTURE=9,
        EYE_START=10,EYE_WAIT=11,RESULT=12,SEND=13,WAIT_SEND=14,DRAIN=15,
        DEBUG_REQ=16,DEBUG_WAIT=17,DEBUG_CAP=18,BLINK_COMMIT=19,TIMING_UPDATE=20,TIMING_REPLY=21,
        OMZ_START=22,OMZ_WAIT=23,OMZ_REPLY=24;
    reg [4:0] state=IDLE;
    reg [7:0] command=0;
    reg [31:0] sequence_id=0;
    reg [15:0] length=0;
    reg [10:0] position=0;
    reg [7:0] active_method=PRETRAINED ? 8'd4 : DEFAULT_METHOD,new_method=0;
    reg [31:0] fingerprint=PRETRAINED ? OMZ_FINGERPRINT : DEFAULT_FINGERPRINT,new_fingerprint=0;
    reg model_loaded=PRETRAINED;
    reg [12:0] uploaded=0;
    reg [15:0] upload_offset=0;
    reg signed [31:0] left_threshold=0,right_threshold=0;
    reg [8:0] pixel_index=0;
    reg eye_select=0;
    reg [2:0] input_offset=1;
    reg signed [31:0] left_open=0,left_closed=0,right_open=0,right_closed=0;
    reg [31:0] left_cycles=0,right_cycles=0,pair_elapsed=0;
    wire signed [32:0] left_margin=$signed({left_closed[31],left_closed})-$signed({left_open[31],left_open});
    wire signed [32:0] right_margin=$signed({right_closed[31],right_closed})-$signed({right_open[31],right_open});
    assign decisions={right_margin>$signed({right_threshold[31],right_threshold}),
                          left_margin>$signed({left_threshold[31],left_threshold}),
                          right_margin>0,left_margin>0};
    assign clear_blinks=reset || (state==DISPATCH && command==8'h13 && length==0);
    assign invalidate_blinks=state==MODEL_COMMIT || state==CONFIG_COMMIT ||
        (state==DISPATCH && command==8'h11 && length==0);
    assign start_pair=state==READ_CAP &&
        ((command==8'h10 && position==0 && read_data==active_method) ||
         (command==8'h12 && position==4));
    assign pair_active=state>=EYE_REQ && state<=EYE_WAIT;
    assign result_update=state==RESULT;
    assign timed_result=command==8'h12;
    assign timing_update=state==TIMING_UPDATE;
    assign timing_reply=state==TIMING_REPLY;
    assign reply_kind=command|8'h80;
    assign reply_sequence=sequence_id;
    reg [7:0] reply[0:63];
    assign reply_data=reply[reply_address];
    wire [1:0] pixel_code= pixel_index[1:0]==0 ? read_data[7:6] :
                           pixel_index[1:0]==1 ? read_data[5:4] :
                           pixel_index[1:0]==2 ? read_data[3:2] : read_data[1:0];
    integer i;
    initial for(i=0;i<64;i=i+1) reply[i]=0;
    always @(posedge clk) begin
        rx_accept<=0;model_we<=0;input_we<=0;core_start<=0;tx_start<=0;
        if(key_pulse[0])page<=page+1'b1;
        if(key_pulse[1])page<=page-1'b1;
        if(key_pulse[2])frames<=0;
        if(state>=EYE_REQ && state<=EYE_WAIT)pair_elapsed<=pair_elapsed+1'b1;
        if(reset)state<=IDLE;
        else case(state)
            IDLE:if(packet_valid)begin
                command<=packet_kind;sequence_id<=packet_sequence;length<=packet_length;state<=DISPATCH;
            end
            DISPATCH:begin
                for(i=0;i<64;i=i+1)reply[i]<=0;
                reply_length<=1;
                if(command==1 && length==0)begin
                    reply_length<=12;reply[1]<=active_method;reply[2]<={7'd0,model_loaded};reply[3]<={2'b11,4'd0,page};
                    reply[4]<=8'h41;reply[5]<=8'h45;reply[6]<=8'h45;reply[7]<=8'h45;
                    for(i=0;i<4;i=i+1)reply[8+i]<=fingerprint>>(i*8);
                    state<=SEND;
                end else if((!PRETRAINED && command==2 && length==1213) || (command==3 && length==8) ||
                    (!PRETRAINED && command==5 && length==2) || (command==8'h14 && length==9) ||
                    (PRETRAINED && command==8'h20 && length>2 && length<=1026))begin
                    position<=0;state<=READ_REQ;
                end else if(PRETRAINED && command==8'h21 && length==0 && uploaded==3072)begin
                    uploaded<=0;state<=OMZ_START;
                end else if(!PRETRAINED && command==8'h10 && model_loaded && length==(active_method==0 ? 1025 : 257))begin
                    position<=0;state<=READ_REQ;
                end else if(!PRETRAINED && command==8'h12 && model_loaded && length==(active_method==0 ? 1029 : 261))begin
                    position<=0;state<=READ_REQ;
                end else if(command==8'h11 && length==0)begin result_valid<=0;state<=SEND;end
                else if(command==8'h13 && length==0)state<=SEND;
                else begin reply[0]<=2;state<=SEND;end
            end
            READ_REQ:begin read_address<=position;state<=READ_WAIT;end
            READ_WAIT:state<=READ_CAP;
            READ_CAP:begin
                if(command==8'h20)begin
                    if(position==0)begin upload_offset[7:0]<=read_data;position<=1;state<=READ_REQ;end
                    else if(position==1)begin
                        upload_offset[15:8]<=read_data;
                        // Offset zero starts a fresh eye. Other chunks must be contiguous.
                        if(({read_data,upload_offset[7:0]}!=0 && {read_data,upload_offset[7:0]}!=uploaded) ||
                            {read_data,upload_offset[7:0]}+length-2>3072)begin
                            uploaded<=0;reply[0]<=5;state<=SEND;
                        end else begin position<=2;state<=READ_REQ;end
                    end else begin
                        input_we<=1;input_address<=upload_offset+position-2;input_data<=read_data;
                        if(position==length-1)begin uploaded<=upload_offset+length-2;state<=SEND;end
                        else begin position<=position+1'b1;state<=READ_REQ;end
                    end
                end else if(command==8'h14)begin
                    if(position<4)capture_ms[position*8 +: 8]<=read_data;
                    else if(position==4)timing_valid<=read_data[1:0];
                    else if(position<7)left_score[(position-5)*8 +: 8]<=read_data;
                    else right_score[(position-7)*8 +: 8]<=read_data;
                    if(position==8)begin timing_mode<=1;state<=TIMING_UPDATE;end
                    else begin position<=position+1'b1;state<=READ_REQ;end
                end else if(command==2)begin
                    if(position==0)new_method<=read_data;
                    else if(position<=4)new_fingerprint[(position-1)*8 +: 8]<=read_data;
                    else begin model_we<=1;model_address<=position-5;model_data<=read_data;end
                    if(position==1212)state<=MODEL_COMMIT;
                    else begin position<=position+1'b1;state<=READ_REQ;end
                end else if(command==3)begin
                    if(position<4)left_threshold[position*8 +: 8]<=read_data;
                    else right_threshold[(position-4)*8 +: 8]<=read_data;
                    if(position==7)state<=CONFIG_COMMIT;
                    else begin position<=position+1'b1;state<=READ_REQ;end
                end else if(command==5)begin
                    if(position==0)begin debug_address[7:0]<=read_data;position<=1;state<=READ_REQ;end
                    else begin debug_address[10:8]<=read_data[2:0];position<=0;reply_length<=17;state<=DEBUG_REQ;end
                end else if(position==0)begin
                    if(read_data!=active_method)begin reply[0]<=4;state<=SEND;end
                    else if(command==8'h12)begin input_offset<=5;position<=1;state<=READ_REQ;end
                    else begin
                        input_offset<=1;eye_select<=0;pixel_index<=0;pair_elapsed<=0;state<=EYE_REQ;
                    end
                end else begin
                    capture_ms[(position-1)*8 +: 8]<=read_data;
                    if(position==4)begin eye_select<=0;pixel_index<=0;pair_elapsed<=0;state<=EYE_REQ;end
                    else begin position<=position+1'b1;state<=READ_REQ;end
                end
            end
            MODEL_COMMIT:begin
                model_loaded<=(new_method<=3);
                if(new_method<=3)begin active_method<=new_method;fingerprint<=new_fingerprint;end
                else begin reply[0]<=3;end
                result_valid<=0;state<=SEND;
            end
            CONFIG_COMMIT:state<=SEND;
            OMZ_START:begin core_start<=1;state<=OMZ_WAIT;end
            OMZ_WAIT:if(core_done)state<=OMZ_REPLY;
            OMZ_REPLY:begin
                reply_length<=17;
                for(i=0;i<4;i=i+1)begin
                    reply[1+i]<=core_open>>(i*8);reply[5+i]<=core_closed>>(i*8);
                    reply[9+i]<=core_cycles>>(i*8);reply[13+i]<=fingerprint>>(i*8);
                end
                state<=SEND;
            end
            DEBUG_REQ:state<=DEBUG_WAIT;
            DEBUG_WAIT:state<=DEBUG_CAP;
            DEBUG_CAP:begin
                reply[1+position]<=debug_data;
                if(position==15)state<=SEND;
                else begin position<=position+1'b1;debug_address<=debug_address+1'b1;state<=DEBUG_REQ;end
            end
            EYE_REQ:begin
                read_address<=input_offset+(eye_select ? (active_method==0 ? 512 : 128) : 0)+
                    (active_method==0 ? {2'd0,pixel_index} : {4'd0,pixel_index[8:2]});
                state<=EYE_WAIT_READ;
            end
            EYE_WAIT_READ:state<=EYE_CAPTURE;
            EYE_CAPTURE:begin
                input_we<=1;input_address<=pixel_index;
                input_data<=active_method==0 ? read_data : {pixel_code,pixel_code,pixel_code,pixel_code};
                if(pixel_index==511)state<=EYE_START;
                else begin pixel_index<=pixel_index+1'b1;state<=EYE_REQ;end
            end
            EYE_START:begin core_start<=1;state<=EYE_WAIT;end
            EYE_WAIT:if(core_done)begin
                if(!eye_select)begin
                    left_open<=core_open;left_closed<=core_closed;left_cycles<=core_cycles;
                    eye_select<=1;pixel_index<=0;state<=EYE_REQ;
                end else begin
                    right_open<=core_open;right_closed<=core_closed;right_cycles<=core_cycles;state<=RESULT;
                end
            end
            RESULT:begin
                timing_mode<=0;
                reply_length<=32;reply[1]<=active_method;reply[2]<={4'd0,decisions};
                for(i=0;i<4;i=i+1)begin
                    reply[4+i]<=left_open>>(i*8);reply[8+i]<=left_closed>>(i*8);
                    reply[12+i]<=right_open>>(i*8);reply[16+i]<=right_closed>>(i*8);
                    reply[20+i]<=left_cycles>>(i*8);reply[24+i]<=right_cycles>>(i*8);reply[28+i]<=pair_elapsed>>(i*8);
                end
                result_flags<=decisions;result_valid<=1;frames<=frames+1'b1;last_pair_cycles<=pair_elapsed[23:0];
                state<=command==8'h12 ? BLINK_COMMIT : SEND;
            end
            BLINK_COMMIT:begin
                reply_length<=40;reply[3]<={6'd0,blink_events};
                for(i=0;i<4;i=i+1)begin
                    reply[32+i]<=left_blinks>>(i*8);reply[36+i]<=right_blinks>>(i*8);
                end
                state<=SEND;
            end
            TIMING_UPDATE:state<=TIMING_REPLY;
            TIMING_REPLY:begin
                reply_length<=36;reply[1]<=1;reply[2]<={6'd0,timing_closed};
                reply[3]<={timing_event,5'd0,timing_known};
                for(i=0;i<4;i=i+1)begin
                    reply[4+i]<=timing_count>>(i*8);reply[8+i]<=timing_long>>(i*8);
                    reply[12+i]<=left_current_ms>>(i*8);reply[16+i]<=right_current_ms>>(i*8);
                    reply[20+i]<=last_left_ms>>(i*8);reply[24+i]<=last_right_ms>>(i*8);
                    reply[28+i]<=last_overlap_ms>>(i*8);reply[32+i]<=last_duration_ms>>(i*8);
                end
                result_flags<={timing_closed,timing_closed};result_valid<=timing_known==3;
                frames<=frames+1'b1;state<=SEND;
            end
            SEND:begin tx_start<=1;state<=WAIT_SEND;end
            WAIT_SEND:if(tx_packet_done)begin rx_accept<=1;state<=DRAIN;end
            DRAIN:if(!packet_valid)state<=IDLE;
        endcase
    end
endmodule
