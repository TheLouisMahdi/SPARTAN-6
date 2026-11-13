// TinyEye: valid conv 1->4, pool, valid conv 4->8, pool, FC96->8, FC8->2.
// Pooling is fused: compute four required convolutions per pooled output.
// Activations store magnitudes q+128; arithmetic matches integer_model.py.
module eye_cnn(input wire clk, input wire reset, input wire start,
    input wire model_we, input wire [10:0] model_address, input wire [7:0] model_data,
    input wire input_we, input wire [8:0] input_address, input wire [7:0] input_data,
    output reg busy=0, output reg done=0, output reg signed [31:0] logit_open=0,
    output reg signed [31:0] logit_closed=0, output reg [31:0] cycles=0,
    input wire [10:0] debug_address, output wire [7:0] debug_data);
    localparam IDLE=0, BIAS_REQ=1, BIAS_WAIT=2, BIAS_CAP=3,
        MULT_REQ=4, MULT_WAIT=5, MULT_CAP=6, DOT_REQ=7, DOT_WAIT=8,
        DOT_MAC=9, RQ_START=10, RQ_WAIT=11, OUTPUT=12, ADVANCE=13;
    reg [3:0] state=IDLE;
    reg [1:0] layer=0, pool_cell=0, byte_index=0;
    reg [3:0] out_channel=0, out_x=0;
    reg [2:0] out_y=0, in_channel=0;
    reg [1:0] kernel_x=0, kernel_y=0;
    reg [6:0] feature=0;
    reg [8:0] output_index=0;
    reg [31:0] assembled=0, multiplier=0, elapsed=0;
    reg signed [31:0] accumulator=0;
    reg [7:0] pool_max=0, output_value=0;
    reg [10:0] weight_address;
    wire [8:0] activation_address;
    wire [10:0] coefficient_address;
    wire [4:0] bias_base;
    wire [7:0] weight_data, a_data, b_data;
    wire [8:0] activation_read_address = !busy && debug_address>=1280 ?
        (debug_address>=1536 ? debug_address[8:0] : {1'b0,debug_address[7:0]}) : activation_address;
    wire signed [31:0] next_accumulator;
    wire [7:0] rq_value;
    wire rq_done, rq_busy;
    wire [7:0] maximum = rq_value>pool_max ? rq_value : pool_max;
    wire a_write = (!busy && input_we) || (state==OUTPUT && layer==1);
    wire b_write = (state==OUTPUT && (layer==0 || layer==2));
    wire [8:0] a_write_address = busy ? output_index : input_address;
    wire [7:0] a_write_data = busy ? output_value : input_data;
    wire dot_last = layer==0 ? (kernel_x==2 && kernel_y==2) :
                    layer==1 ? (kernel_x==2 && kernel_y==2 && in_channel==3) :
                    layer==2 ? (feature==95) : (feature==7);
    cnn_addresses addresses(layer,out_channel,out_x,out_y,in_channel,kernel_x,kernel_y,
                            pool_cell,feature,coefficient_address,activation_address,bias_base);
    always @* begin
        weight_address=coefficient_address;
        if (state==BIAS_REQ || state==BIAS_WAIT || state==BIAS_CAP)
            weight_address=1108+(bias_base+out_channel)*4+byte_index;
        else if (state==MULT_REQ || state==MULT_WAIT || state==MULT_CAP)
            weight_address=1196+layer*4+byte_index;
    end
    model_store models(clk,model_we && !busy,model_address,model_data,busy ? weight_address : debug_address,weight_data);
    assign debug_data=debug_address>=1536 ? a_data : debug_address>=1280 ? b_data : weight_data;
    activation_store bank_a(clk,a_write,a_write_address,a_write_data,activation_read_address,a_data);
    activation_store bank_b(clk,b_write,output_index,output_value,activation_read_address,b_data);
    mac_int8 mac(layer[0] ? b_data : a_data,weight_data,accumulator,next_accumulator);
    requantize_q24 rq(clk,reset,state==RQ_START,accumulator,multiplier,rq_busy,rq_done,rq_value);
    always @(posedge clk) begin
        done<=0;
        if (busy) elapsed<=elapsed+1'b1;
        if (reset) begin state<=IDLE; busy<=0; end
        else case (state)
            IDLE: if (start) begin
                busy<=1; elapsed<=0; layer<=0; out_channel<=0; out_x<=0; out_y<=0;
                pool_cell<=0; pool_max<=0; byte_index<=0; output_index<=0;
                feature<=0; kernel_x<=0; kernel_y<=0; in_channel<=0; state<=BIAS_REQ;
            end
            BIAS_REQ: state<=BIAS_WAIT;
            BIAS_WAIT: state<=BIAS_CAP;
            BIAS_CAP: begin
                assembled[byte_index*8 +: 8]<=weight_data;
                if (byte_index==3) begin
                    accumulator<={weight_data,assembled[23:0]}; byte_index<=0;
                    state<=layer==3 ? DOT_REQ : MULT_REQ;
                end else begin byte_index<=byte_index+1'b1; state<=BIAS_REQ; end
            end
            MULT_REQ: state<=MULT_WAIT;
            MULT_WAIT: state<=MULT_CAP;
            MULT_CAP: begin
                assembled[byte_index*8 +: 8]<=weight_data;
                if (byte_index==3) begin multiplier<={weight_data,assembled[23:0]}; byte_index<=0; state<=DOT_REQ; end
                else begin byte_index<=byte_index+1'b1; state<=MULT_REQ; end
            end
            DOT_REQ: state<=DOT_WAIT;
            DOT_WAIT: state<=DOT_MAC;
            DOT_MAC: begin
                accumulator<=next_accumulator;
                if (dot_last) begin
                    kernel_x<=0; kernel_y<=0; in_channel<=0; feature<=0;
                    state<=layer==3 ? OUTPUT : RQ_START;
                end else begin
                    if (layer>=2) feature<=feature+1'b1;
                    else if (kernel_x==2) begin
                        kernel_x<=0;
                        if (kernel_y==2) begin kernel_y<=0; in_channel<=in_channel+1'b1; end
                        else kernel_y<=kernel_y+1'b1;
                    end else kernel_x<=kernel_x+1'b1;
                    state<=DOT_REQ;
                end
            end
            RQ_START: state<=RQ_WAIT;
            RQ_WAIT: if (rq_done) begin
                if (layer==2) begin output_value<=rq_value; state<=OUTPUT; end
                else if (pool_cell==3) begin output_value<=maximum; pool_max<=0; pool_cell<=0; state<=OUTPUT; end
                else begin pool_max<=maximum; pool_cell<=pool_cell+1'b1; state<=BIAS_REQ; end
            end
            OUTPUT: begin
                if (layer==3) begin
                    if (out_channel==0) logit_open<=accumulator;
                    else logit_closed<=accumulator;
                end
                state<=ADVANCE;
            end
            ADVANCE: begin
                if (layer==3 && out_channel==1) begin
                    busy<=0; done<=1; cycles<=elapsed+1'b1; state<=IDLE;
                end else if ((layer==0 && out_channel==3 && out_y==6 && out_x==14) ||
                             (layer==1 && out_channel==7 && out_y==1 && out_x==5) ||
                             (layer==2 && out_channel==7)) begin
                    layer<=layer+1'b1; out_channel<=0; out_x<=0; out_y<=0; output_index<=0; state<=BIAS_REQ;
                end else begin
                    output_index<=output_index+1'b1;
                    if (layer>=2) out_channel<=out_channel+1'b1;
                    else if (out_x==(layer==0 ? 14 : 5)) begin
                        out_x<=0;
                        if (out_y==(layer==0 ? 6 : 1)) begin out_y<=0; out_channel<=out_channel+1'b1; end
                        else out_y<=out_y+1'b1;
                    end else out_x<=out_x+1'b1;
                    state<=BIAS_REQ;
                end
            end
        endcase
    end
endmodule
