// Optional compact-build implementation. Same positive unsigned product,
// Q24 half-up rounding and 32 busy cycles as the serial reference.
module requantize_q24_dsp(input wire clk,input wire reset,input wire start,
    input wire signed [31:0] accumulator,input wire [31:0] multiplier,
    output reg busy=0,output reg done=0,output reg [7:0] magnitude=0);
    reg [63:0] product=0;
    reg [5:0] step=0;
    (* use_dsp48="yes" *) wire [63:0] multiplied=$unsigned(accumulator)*multiplier;
    wire [63:0] rounded=product+64'h800000;
    always @(posedge clk)begin
        done<=0;
        if(reset)busy<=0;
        else if(start&&!busy)begin
            if(accumulator<=0)begin magnitude<=0;done<=1;end
            else begin product<=multiplied;step<=0;busy<=1;end
        end else if(busy)begin
            if(step==31)begin
                magnitude<=(rounded>>24)>255 ? 8'd255 : rounded[31:24];
                busy<=0;done<=1;
            end else step<=step+1'b1;
        end
    end
endmodule
