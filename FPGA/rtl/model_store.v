// 1108 signed weights, 22 little-endian INT32 biases, three uint32 multipliers.
module model_store(input wire clk, input wire write_enable, input wire [10:0] write_address,
    input wire [7:0] write_data, input wire [10:0] read_address, output reg [7:0] read_data);
    (* ram_style = "block" *) reg [7:0] memory [0:2047];
    initial $readmemh("generated/model.hex", memory);
    always @(posedge clk) begin
        if (write_enable) memory[write_address] <= write_data;
        read_data <= memory[read_address];
    end
endmodule
