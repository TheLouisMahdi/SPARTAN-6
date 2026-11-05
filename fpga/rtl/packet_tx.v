module packet_tx(input wire clk, input wire reset, input wire start,
    input wire [7:0] kind, input wire [31:0] sequence_id, input wire [5:0] length,
    output wire [5:0] payload_address, input wire [7:0] payload_data,
    output reg byte_start=0, output reg [7:0] byte_data=0, input wire byte_busy,
    output wire busy, output reg done=0);
    `include "rtl/crc16_function.vh"
    reg [1:0] state=0;
    reg [7:0] position=0, saved_kind=0;
    reg [31:0] saved_sequence=0;
    reg [5:0] saved_length=0;
    reg [15:0] crc=16'hffff;
    reg [7:0] current;
    assign busy=(state!=0);
    assign payload_address=position-10;
    always @* begin
        case (position)
            0:current=8'ha5;
            1:current=8'h5a;
            2:current=1;
            3:current=saved_kind;
            4:current=saved_sequence[7:0];
            5:current=saved_sequence[15:8];
            6:current=saved_sequence[23:16];
            7:current=saved_sequence[31:24];
            8:current={2'd0,saved_length};
            9:current=0;
            default: if (position==10+saved_length) current=crc[7:0];
                     else if (position==11+saved_length) current=crc[15:8];
                     else current=payload_data;
        endcase
    end
    always @(posedge clk) begin
        byte_start<=0; done<=0;
        if (reset) state<=0;
        else case (state)
            0: if (start) begin
                saved_kind<=kind; saved_sequence<=sequence_id; saved_length<=length;
                position<=0; crc<=16'hffff; state<=1;
            end
            1: if (!byte_busy) begin
                byte_data<=current; byte_start<=1; state<=2;
                if (position>=2 && position<10+saved_length) crc<=crc16_byte(crc,current);
            end
            2: if (byte_busy) state<=3;
            3: if (!byte_busy) begin
                if (position==11+saved_length) begin state<=0; done<=1; end
                else begin position<=position+1'b1; state<=1; end
            end
        endcase
    end
endmodule
