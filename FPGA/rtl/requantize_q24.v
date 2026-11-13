// Exact unsigned shift/add multiplication avoids a large parallel multiplier.
// ReLU, half-up Q24 rounding, saturation to the activation magnitude 0..255.
module requantize_q24(input wire clk, input wire reset, input wire start,
    input wire signed [31:0] accumulator, input wire [31:0] multiplier,
    output reg busy = 0, output reg done = 0, output reg [7:0] magnitude = 0);
    reg [63:0] product = 0, multiplicand = 0;
    reg [31:0] remaining = 0;
    reg [5:0] step = 0;
    wire [63:0] added = product + (remaining[0] ? multiplicand : 64'd0);
    wire [63:0] rounded = added + 64'h800000;
    always @(posedge clk) begin
        done <= 0;
        if (reset) busy <= 0;
        else if (start && !busy) begin
            if (accumulator <= 0) begin magnitude <= 0; done <= 1; end
            else begin
                product <= 0; multiplicand <= {32'd0, accumulator};
                remaining <= multiplier; step <= 0; busy <= 1;
            end
        end else if (busy) begin
            product <= added; multiplicand <= multiplicand << 1; remaining <= remaining >> 1;
            if (step == 31) begin
                magnitude <= (rounded >> 24) > 255 ? 8'd255 : rounded[31:24];
                busy <= 0; done <= 1;
            end else step <= step + 1'b1;
        end
    end
endmodule
