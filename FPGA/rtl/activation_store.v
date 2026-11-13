// Two independent synchronous ports: host input/writeback and CNN reads.
module activation_store(input wire clk, input wire write_enable,
    input wire [8:0] write_address, input wire [7:0] write_data,
    input wire [8:0] read_address, output reg [7:0] read_data);
    (* ram_style = "block" *) reg [7:0] memory [0:511];
    always @(posedge clk) begin
        if (write_enable) memory[write_address] <= write_data;
        read_data <= memory[read_address];
    end
endmodule
