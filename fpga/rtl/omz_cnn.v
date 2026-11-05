// Pinned open-closed-eye-0001. Two output channels share each activation.
// INT16 coefficients, INT32 activations, INT48 MAC; no retraining.
module omz_cnn #(parameter integer PE_COUNT=2)(
    input wire clk, input wire reset, input wire start,
    input wire input_we, input wire [11:0] input_address, input wire [7:0] input_data,
    output reg busy=0, output reg done=0,
    output reg signed [31:0] logit_open=0, output reg signed [31:0] logit_closed=0,
    output reg [31:0] cycles=0,
    input wire [11:0] debug_address, output wire signed [31:0] debug_data);
    `include "generated/omz_config.vh"
    localparam IDLE=0,BIAS_REQ=1,BIAS_WAIT=2,BIAS_CAP=3,
        DOT_REQ=4,DOT_WAIT=5,DOT_MAC=6,QUANTIZE=7,WRITE0=8,WRITE1=9,ADVANCE=10;
    reg [3:0] state=IDLE;
    reg [1:0] layer=0,pool_cell=0;
    reg [5:0] channel=0,in_channel=0;
    reg [3:0] x=0,y=0;
    reg [1:0] kx=0,ky=0;
    reg [8:0] coefficient=0;
    reg [13:0] weight_base=0;
    reg [11:0] output_index=0;
    reg [31:0] elapsed=0;
    reg read_valid=0,product_valid=0;
    reg signed [47:0] accumulator[0:PE_COUNT-1],product[0:PE_COUNT-1];
    reg signed [31:0] maximum[0:PE_COUNT-1],value[0:PE_COUNT-1];
    reg signed [31:0] global_max[0:1];
    // Explicit banks avoid wasting four BRAMs on power-of-two depth rounding.
    wire [13:0] weight_address[0:PE_COUNT-1];
    wire [PE_COUNT*16-1:0] rom_data[0:2];
    reg [1:0] weight_bank[0:PE_COUNT-1];
    // XCN11014: never depend on configuration-time initialization of 9K BRAM.
    // This small ROM uses LUTs; coefficient ROMs use full 18K blocks.
    (* rom_style="distributed" *) reg signed [47:0] biases[0:127];
    wire [8:0] dot_length=layer==0 ? 27 : layer==1 ? 90 : layer==2 ? 180 : 50;
    wire [7:0] plane=layer==0 ? 225 : layer==1 ? 36 : 16;
    wire [3:0] last_xy=layer==0 ? 14 : layer==1 ? 5 : 3;
    wire [5:0] last_channel=layer==0 ? 10-PE_COUNT : layer==1 ? 20-PE_COUNT :
        layer==2 ? 50-PE_COUNT : 2-PE_COUNT;
    wire [6:0] bias_base=layer==0 ? 0 : layer==1 ? 10 : layer==2 ? 30 : 80;
    reg [11:0] address;
    always @* begin
        case(layer)
            0:address={in_channel[1:0],10'b0}+
                (({8'd0,y,1'b0}+pool_cell[1]+ky)<<5)+{7'd0,x,1'b0}+pool_cell[0]+kx;
            1:address=in_channel*12'd225+
                ({7'd0,y,1'b0}+pool_cell[1]+ky)*12'd15+{7'd0,x,1'b0}+pool_cell[0]+kx;
            2:address=in_channel*12'd36+({8'd0,y}+ky)*12'd6+{8'd0,x}+kx;
            3:address={in_channel,4'b0}+{6'd0,y,2'b0}+x;
        endcase
    end
    (* ram_style="block" *) reg signed [31:0] bank_a[0:3071];
    (* ram_style="block" *) reg signed [31:0] bank_b[0:2249];
    reg signed [31:0] a_data=0,b_data=0;
    wire [11:0] read_address=busy ? address : debug_address;
    wire write_output=state==WRITE0 || state==WRITE1;
    wire [11:0] write_address=output_index+(state==WRITE1 ? {4'd0,plane} : 12'd0);
    wire signed [31:0] write_value=state==WRITE1 ? value[PE_COUNT-1] : value[0];
    wire signed [9:0] centered=$signed({1'b0,input_data})-10'sd127;
    wire signed [31:0] normalized=(centered*32'sd257)+(input_data==255 ? 32'sd1 : 32'sd0);
    wire signed [31:0] activation=layer[0] ? b_data : a_data;
    assign debug_data=layer[0] ? a_data : b_data;
    initial begin
        $readmemh("generated/omz_biases.hex",biases);
        global_max[0]=32'sh80000000;global_max[1]=32'sh80000000;
    end
    always @(posedge clk)begin
        if(input_we && !busy && input_address<3072)bank_a[input_address]<=normalized;
        else if(write_output && layer==1)bank_a[write_address]<=write_value;
        if(write_output && (layer==0 || layer==2))bank_b[write_address]<=write_value;
        a_data<=bank_a[read_address];
        b_data<=bank_b[read_address];
    end
    // One synchronous read, registered multiply and accumulator update overlap.
    always @(posedge clk)begin
        if(reset)begin read_valid<=0;product_valid<=0;end
        else begin
            read_valid<=state==DOT_REQ;
            product_valid<=read_valid;
        end
    end
    genvar bank,reader,lane;
    generate for(bank=0;bank<3;bank=bank+1)begin: rom
        (* rom_style="block" *) reg signed [15:0] memory[0:4095];
        if(bank==0)initial $readmemh("generated/omz_weights0.hex",memory);
        if(bank==1)initial $readmemh("generated/omz_weights1.hex",memory);
        if(bank==2)initial $readmemh("generated/omz_weights2.hex",memory);
        for(reader=0;reader<PE_COUNT;reader=reader+1)begin: port
            reg signed [15:0] data=0;
            always @(posedge clk)data<=memory[weight_address[reader][11:0]];
            assign rom_data[bank][reader*16 +: 16]=data;
        end
    end endgenerate
    generate for(lane=0;lane<PE_COUNT;lane=lane+1)begin: pe
        wire signed [15:0] weight=weight_bank[lane]==0 ? rom_data[0][lane*16 +: 16] :
            weight_bank[lane]==1 ? rom_data[1][lane*16 +: 16] : rom_data[2][lane*16 +: 16];
        reg signed [47:0] bias=0;
        wire [6:0] bias_address=bias_base+{1'b0,channel}+lane;
        assign weight_address[lane]=weight_base+coefficient+lane*dot_length;
        wire signed [47:0] rounded=accumulator[lane]+(48'sd1<<(omz_shift(layer)-1'b1));
        wire signed [47:0] shifted=rounded>>>omz_shift(layer);
        wire signed [31:0] quantized=shifted[31:0];
        wire signed [31:0] pooled=quantized>maximum[lane] ? quantized : maximum[lane];
        always @(posedge clk)begin
            weight_bank[lane]<=weight_address[lane][13:12];
            bias<=biases[bias_address];
            if(read_valid)product[lane]<=activation*weight;
            if(state==BIAS_CAP)accumulator[lane]<=bias;
            else if(product_valid)accumulator[lane]<=accumulator[lane]+product[lane];
            if(state==IDLE && start)maximum[lane]<=32'sh80000000;
            if(state==QUANTIZE)begin
                if(layer<2)begin
                    if(pool_cell==3)begin
                        value[lane]<=pooled<0 ? 32'sd0 : pooled;
                        maximum[lane]<=32'sh80000000;
                    end else maximum[lane]<=pooled;
                end else value[lane]<=quantized;
            end
        end
    end endgenerate
    integer i;
    always @(posedge clk)begin
        done<=0;
        if(busy)elapsed<=elapsed+1'b1;
        if(reset)begin busy<=0;state<=IDLE;end
        else case(state)
            IDLE:if(start)begin
                busy<=1;elapsed<=0;layer<=0;channel<=0;in_channel<=0;x<=0;y<=0;
                kx<=0;ky<=0;coefficient<=0;pool_cell<=0;output_index<=0;weight_base<=0;
                global_max[0]<=32'sh80000000;global_max[1]<=32'sh80000000;
                state<=BIAS_REQ;
            end
            BIAS_REQ:state<=BIAS_WAIT;
            BIAS_WAIT:state<=BIAS_CAP;
            BIAS_CAP:state<=DOT_REQ;
            DOT_REQ:begin
                if(coefficient==dot_length-1'b1)begin
                    state<=DOT_WAIT;
                end else begin
                    coefficient<=coefficient+1'b1;
                    if(layer==3)in_channel<=in_channel+1'b1;
                    else if(kx==2)begin
                        kx<=0;
                        if(ky==2)begin ky<=0;in_channel<=in_channel+1'b1;end
                        else ky<=ky+1'b1;
                    end else kx<=kx+1'b1;
                end
            end
            // Drain the final memory read and product before requantization.
            DOT_WAIT:state<=DOT_MAC;
            DOT_MAC:begin
                coefficient<=0;in_channel<=0;kx<=0;ky<=0;state<=QUANTIZE;
            end
            QUANTIZE:begin
                if(layer>=2 || pool_cell==3)begin pool_cell<=0;state<=WRITE0;end
                else begin pool_cell<=pool_cell+1'b1;state<=BIAS_REQ;end
            end
            WRITE0:begin
                if(layer==3 && value[0]>global_max[channel])global_max[channel]<=value[0];
                state<=PE_COUNT==2 ? WRITE1 : ADVANCE;
            end
            WRITE1:begin
                if(layer==3 && value[PE_COUNT-1]>global_max[1])global_max[1]<=value[PE_COUNT-1];
                state<=ADVANCE;
            end
            ADVANCE:begin
                state<=BIAS_REQ;
                if(x==last_xy && y==last_xy)begin
                    x<=0;y<=0;
                    if(channel==last_channel)begin
                        channel<=0;output_index<=0;
                        if(layer==3)begin
                            logit_open<=global_max[0];logit_closed<=global_max[1];
                            busy<=0;done<=1;cycles<=elapsed+1'b1;state<=IDLE;
                        end else begin
                            layer<=layer+1'b1;
                            weight_base<=layer==0 ? 14'd270 : layer==1 ? 14'd2070 : 14'd11070;
                        end
                    end else begin
                        channel<=channel+PE_COUNT;
                        weight_base<=weight_base+PE_COUNT*dot_length;
                        output_index<=output_index+(PE_COUNT-1)*plane+1'b1;
                    end
                end else begin
                    output_index<=output_index+1'b1;
                    if(x==last_xy)begin x<=0;y<=y+1'b1;end
                    else x<=x+1'b1;
                end
            end
        endcase
    end
endmodule
