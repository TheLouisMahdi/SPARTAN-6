// Two output channels share one activation read and use replicated weight RAM.
// Same tensor order, Q24 arithmetic and debug map as eye_cnn (the P1 core).
module eye_cnn_parallel(input wire clk, input wire reset, input wire start,
    input wire model_we, input wire [10:0] model_address, input wire [7:0] model_data,
    input wire input_we, input wire [8:0] input_address, input wire [7:0] input_data,
    output reg busy=0, output reg done=0, output reg signed [31:0] logit_open=0,
    output reg signed [31:0] logit_closed=0, output reg [31:0] cycles=0,
    input wire [10:0] debug_address, output wire [7:0] debug_data);
    localparam IDLE=0, BIAS_REQ=1, BIAS_WAIT=2, BIAS_CAP=3,
        MULT_REQ=4, MULT_WAIT=5, MULT_CAP=6, DOT_REQ=7, DOT_WAIT=8,
        DOT_MAC=9, RQ_START=10, RQ_WAIT=11, OUTPUT0=12, OUTPUT1=13, ADVANCE=14;
    reg [3:0] state=IDLE;
    reg [1:0] layer=0, pool_cell=0, byte_index=0;
    reg [3:0] out_channel=0, out_x=0;
    reg [2:0] out_y=0, in_channel=0;
    reg [1:0] kernel_x=0, kernel_y=0;
    reg [6:0] feature=0;
    reg [8:0] output_index=0;
    reg [31:0] elapsed=0, multiplier=0, assembled_multiplier=0;
    reg [31:0] assembled_bias[0:1];
    reg signed [31:0] accumulator[0:1];
    reg [7:0] pool_max[0:1], output_value[0:1];
    reg [1:0] rq_finished=0;
    wire [8:0] activation_address[0:1];
    wire [7:0] weight_data[0:1], rq_value[0:1];
    wire signed [31:0] next_accumulator[0:1];
    wire [1:0] rq_done, rq_busy;
    wire [7:0] a_data,b_data;
    wire [8:0] read_address=!busy && debug_address>=1280 ?
        (debug_address>=1536 ? debug_address[8:0] : {1'b0,debug_address[7:0]}) : activation_address[0];
    wire writing=state==OUTPUT0 || state==OUTPUT1;
    wire [8:0] channel_stride=layer==0 ? 9'd105 : layer==1 ? 9'd12 : 9'd1;
    wire [8:0] write_address=output_index+(state==OUTPUT1 ? channel_stride : 9'd0);
    wire [7:0] write_value=state==OUTPUT1 ? output_value[1] : output_value[0];
    wire a_write=(!busy && input_we) || (writing && layer==1);
    wire b_write=writing && (layer==0 || layer==2);
    wire dot_last=layer==0 ? (kernel_x==2 && kernel_y==2) :
        layer==1 ? (kernel_x==2 && kernel_y==2 && in_channel==3) :
        layer==2 ? feature==95 : feature==7;
    // One address generator supplies the broadcast activation and first weight.
    // The second channel's weights are a fixed layer-dependent distance away.
    wire [3:0] channel0={out_channel[3:1],1'b0},channel1={out_channel[3:1],1'b1};
    wire [10:0] coefficient0;
    wire [8:0] activation0;
    wire [4:0] bias0;
    cnn_addresses addresses(layer,channel0,out_x,out_y,in_channel,kernel_x,kernel_y,
        pool_cell,feature,coefficient0,activation0,bias0);
    wire [10:0] weight_stride=layer==0 ? 11'd9 : layer==1 ? 11'd36 : layer==2 ? 11'd96 : 11'd8;
    assign activation_address[0]=activation0;
    assign activation_address[1]=activation0;
    wire bias_cycle=state==BIAS_REQ || state==BIAS_WAIT || state==BIAS_CAP;
    wire multiplier_cycle=state==MULT_REQ || state==MULT_WAIT || state==MULT_CAP;
    wire [5:0] bias_index0={1'b0,bias0}+{2'b0,channel0};
    wire [5:0] bias_index1={1'b0,bias0}+{2'b0,channel1};
    wire [10:0] multiplier_address=11'd1196+{7'd0,layer,2'b00}+{9'd0,byte_index};
    wire [10:0] weight_address0=bias_cycle ? 11'd1108+{3'd0,bias_index0,2'b00}+{9'd0,byte_index} :
        multiplier_cycle ? multiplier_address : coefficient0;
    wire [10:0] weight_address1=bias_cycle ? 11'd1108+{3'd0,bias_index1,2'b00}+{9'd0,byte_index} :
        multiplier_cycle ? multiplier_address : coefficient0+weight_stride;
    genvar lane;
    generate for(lane=0;lane<2;lane=lane+1) begin: pe
        wire [3:0] channel=lane==0 ? channel0 : channel1;
        wire [10:0] weight_address=lane==0 ? weight_address0 : weight_address1;
        model_store models(clk,model_we && !busy,model_address,model_data,
            busy ? weight_address : debug_address,weight_data[lane]);
        mac_int8 mac(layer[0] ? b_data : a_data,weight_data[lane],accumulator[lane],next_accumulator[lane]);
        requantize_q24 rq(clk,reset,state==RQ_START,accumulator[lane],multiplier,
            rq_busy[lane],rq_done[lane],rq_value[lane]);
    end endgenerate
    activation_store bank_a(clk,a_write,busy ? write_address : input_address,
        busy ? write_value : input_data,read_address,a_data);
    activation_store bank_b(clk,b_write,write_address,write_value,read_address,b_data);
    assign debug_data=debug_address>=1536 ? a_data : debug_address>=1280 ? b_data : weight_data[0];
    integer i;
    initial for(i=0;i<2;i=i+1) begin
        assembled_bias[i]=0;accumulator[i]=0;pool_max[i]=0;output_value[i]=0;
    end
    always @(posedge clk) begin
        done<=0;
        if(busy)elapsed<=elapsed+1'b1;
        if(reset)begin state<=IDLE;busy<=0;rq_finished<=0;end
        else case(state)
            IDLE:if(start)begin
                busy<=1;elapsed<=0;layer<=0;out_channel<=0;out_x<=0;out_y<=0;
                pool_cell<=0;byte_index<=0;output_index<=0;feature<=0;
                kernel_x<=0;kernel_y<=0;in_channel<=0;rq_finished<=0;
                for(i=0;i<2;i=i+1)pool_max[i]<=0;
                state<=BIAS_REQ;
            end
            BIAS_REQ:state<=BIAS_WAIT;
            BIAS_WAIT:state<=BIAS_CAP;
            BIAS_CAP:begin
                for(i=0;i<2;i=i+1)begin
                    assembled_bias[i][byte_index*8 +: 8]<=weight_data[i];
                    if(byte_index==3)accumulator[i]<={weight_data[i],assembled_bias[i][23:0]};
                end
                if(byte_index==3)begin byte_index<=0;state<=layer==3 ? DOT_REQ : MULT_REQ;end
                else begin byte_index<=byte_index+1'b1;state<=BIAS_REQ;end
            end
            MULT_REQ:state<=MULT_WAIT;
            MULT_WAIT:state<=MULT_CAP;
            MULT_CAP:begin
                assembled_multiplier[byte_index*8 +: 8]<=weight_data[0];
                if(byte_index==3)begin
                    multiplier<={weight_data[0],assembled_multiplier[23:0]};byte_index<=0;state<=DOT_REQ;
                end else begin byte_index<=byte_index+1'b1;state<=MULT_REQ;end
            end
            DOT_REQ:state<=DOT_WAIT;
            DOT_WAIT:state<=DOT_MAC;
            DOT_MAC:begin
                for(i=0;i<2;i=i+1)accumulator[i]<=next_accumulator[i];
                if(dot_last)begin
                    kernel_x<=0;kernel_y<=0;in_channel<=0;feature<=0;
                    state<=layer==3 ? OUTPUT0 : RQ_START;
                end else begin
                    if(layer>=2)feature<=feature+1'b1;
                    else if(kernel_x==2)begin
                        kernel_x<=0;
                        if(kernel_y==2)begin kernel_y<=0;in_channel<=in_channel+1'b1;end
                        else kernel_y<=kernel_y+1'b1;
                    end else kernel_x<=kernel_x+1'b1;
                    state<=DOT_REQ;
                end
            end
            RQ_START:begin rq_finished<=0;state<=RQ_WAIT;end
            RQ_WAIT:begin
                rq_finished<=rq_finished | rq_done;
                if(&(rq_finished | rq_done))begin
                    for(i=0;i<2;i=i+1)begin
                        if(layer==2)output_value[i]<=rq_value[i];
                        else if(pool_cell==3)begin
                            output_value[i]<=rq_value[i]>pool_max[i] ? rq_value[i] : pool_max[i];pool_max[i]<=0;
                        end else pool_max[i]<=rq_value[i]>pool_max[i] ? rq_value[i] : pool_max[i];
                    end
                    if(layer==2 || pool_cell==3)begin pool_cell<=0;state<=OUTPUT0;end
                    else begin pool_cell<=pool_cell+1'b1;state<=BIAS_REQ;end
                end
            end
            OUTPUT0:begin
                if(layer==3)begin logit_open<=accumulator[0];logit_closed<=accumulator[1];end
                state<=OUTPUT1;
            end
            OUTPUT1:state<=ADVANCE;
            ADVANCE:begin
                if(layer==3)begin busy<=0;done<=1;cycles<=elapsed+1'b1;state<=IDLE;end
                else if((layer==0 && out_channel==2 && out_y==6 && out_x==14) ||
                    (layer==1 && out_channel==6 && out_y==1 && out_x==5) || (layer==2 && out_channel==6))begin
                    layer<=layer+1'b1;out_channel<=0;out_x<=0;out_y<=0;output_index<=0;state<=BIAS_REQ;
                end else begin
                    if(layer>=2)begin out_channel<=out_channel+2;output_index<=output_index+2;end
                    else if(out_x==(layer==0 ? 14 : 5))begin
                        out_x<=0;
                        if(out_y==(layer==0 ? 6 : 1))begin
                            out_y<=0;out_channel<=out_channel+2;output_index<=output_index+channel_stride+1'b1;
                        end else begin out_y<=out_y+1'b1;output_index<=output_index+1'b1;end
                    end else begin out_x<=out_x+1'b1;output_index<=output_index+1'b1;end
                    state<=BIAS_REQ;
                end
            end
        endcase
    end
endmodule
