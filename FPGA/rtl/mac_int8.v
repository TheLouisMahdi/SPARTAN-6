// Nonnegative activation magnitude times signed INT8 weight, INT32 accumulation.
module mac_int8(input wire [7:0] activation, input wire signed [7:0] weight,
                input wire signed [31:0] accumulator, output wire signed [31:0] result);
    wire signed [16:0] product = $signed({1'b0, activation}) * weight;
    assign result = accumulator + {{15{product[16]}}, product};
endmodule
