// Six-digit multiplexing; KEY1/2 select pages, KEY4 holds a lamp test.
module board_display #(parameter integer CLOCK_HZ=50000000)(input wire clk,
    input wire [1:0] page, input wire lamp_test, input wire result_valid,
    input wire [3:0] flags, input wire [23:0] frames, input wire [23:0] cycles,
    input wire [11:0] milliseconds_bcd,
    output reg [7:0] seg=8'hff, output reg [5:0] digit=6'h3f);
    reg [15:0] count=0;
    reg [2:0] selected=0;
    reg blank=1;
    reg [4:0] code;
    reg [6:0] pattern;
    reg [23:0] hex_value;
    always @(posedge clk) begin
        blank<=0;
        if(count==CLOCK_HZ/6000-1) begin
            count<=0; selected<=selected==5 ? 0 : selected+1'b1; blank<=1;
        end else count<=count+1'b1;
    end
    always @* begin
        hex_value=page==2 ? frames : cycles;
        code=16;
        if(page>=2) code={1'b0,((hex_value>>(selected*4)) & 4'hf)};
        else case(selected)
            // digit[0] is physically rightmost on the user's board.
            0:code=result_valid ? {4'd0,(page==0 ? flags[3] : flags[1])} : 17;
            1:code=result_valid ? {4'd0,(page==0 ? flags[2] : flags[0])} : 17;
            2:code=17;
            3:code=result_valid ? {1'b0,milliseconds_bcd[3:0]} : 17;
            4:code=result_valid ? {1'b0,milliseconds_bcd[7:4]} : 17;
            5:code=result_valid ? {1'b0,milliseconds_bcd[11:8]} : 17;
            default:code=16;
        endcase
        case(code)
            0:pattern=7'h3f;1:pattern=7'h06;2:pattern=7'h5b;3:pattern=7'h4f;
            4:pattern=7'h66;5:pattern=7'h6d;6:pattern=7'h7d;7:pattern=7'h07;
            8:pattern=7'h7f;9:pattern=7'h6f;10:pattern=7'h77;11:pattern=7'h7c;
            12:pattern=7'h39;13:pattern=7'h5e;14:pattern=7'h79;15:pattern=7'h71;
            17:pattern=7'h40;18:pattern=7'h38;19:pattern=7'h50;default:pattern=0;
        endcase
        seg=lamp_test ? 8'h00 : {1'b1,~pattern};
        digit=blank ? 6'h3f : ~(6'b000001<<selected);
    end
endmodule
