module uart_rx #(parameter integer CLOCKS_PER_BIT=434)(input wire clk, input wire reset,
    input wire rx, output reg [7:0] data=0, output reg valid=0, output reg framing_error=0);
    (* ASYNC_REG="TRUE" *) reg meta=1, synced=1;
    reg [1:0] state=0;
    reg [15:0] count=0;
    reg [2:0] bit_index=0;
    reg [7:0] shift=0;
    always @(posedge clk) begin
        meta <= rx; synced <= meta;
        valid <= 0; framing_error <= 0;
        if (reset) state <= 0;
        else case (state)
            0: if (!synced) begin state<=1; count<=CLOCKS_PER_BIT/2-1; end
            1: if (count!=0) count<=count-1'b1;
               else if (synced) state<=0;
               else begin state<=2; count<=CLOCKS_PER_BIT-1; bit_index<=0; end
            2: if (count!=0) count<=count-1'b1;
               else begin
                   shift[bit_index]<=synced; count<=CLOCKS_PER_BIT-1;
                   if (bit_index==7) state<=3; else bit_index<=bit_index+1'b1;
               end
            3: if (count!=0) count<=count-1'b1;
               else begin
                   if (synced) begin data<=shift; valid<=1; end
                   else framing_error<=1;
                   state<=0;
               end
        endcase
    end
endmodule
