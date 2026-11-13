// A5 5A, version, kind, sequence LE32, length LE16, payload, CCITT CRC LE16.
module packet_rx #(parameter integer CLOCK_HZ=50000000)(input wire clk, input wire reset,
    input wire [7:0] byte_data, input wire byte_valid, input wire framing_error,
    input wire accept, output reg valid=0, output reg [7:0] kind=0,
    output reg [31:0] sequence_id=0, output reg [15:0] length=0,
    input wire [10:0] read_address, output reg [7:0] read_data,
    output reg [31:0] errors=0);
    `include "rtl/crc16_function.vh"
    (* ram_style="block" *) reg [7:0] payload [0:2047];
    reg [2:0] state=0, header_index=0;
    reg [10:0] position=0;
    reg [15:0] crc=16'hffff;
    reg [7:0] received_crc_low=0;
    reg [23:0] idle_count=0;
    always @(posedge clk) begin
        read_data<=payload[read_address];
        if (reset || accept) begin state<=0; valid<=0; idle_count<=0; end
        else if (!valid) begin
            if (framing_error || (state!=0 && idle_count>=CLOCK_HZ/10)) begin
                state<=0; idle_count<=0; errors<=errors+1'b1;
            end else if (byte_valid) begin
                idle_count<=0;
                case (state)
                    0: if (byte_data==8'ha5) state<=1;
                    1: if (byte_data==8'h5a) begin state<=2; header_index<=0; crc<=16'hffff; end
                       else if (byte_data!=8'ha5) state<=0;
                    2: begin
                        crc<=crc16_byte(crc,byte_data);
                        case (header_index)
                            0: if (byte_data!=1) begin state<=0; errors<=errors+1'b1; end
                            1: kind<=byte_data;
                            2: sequence_id[7:0]<=byte_data;
                            3: sequence_id[15:8]<=byte_data;
                            4: sequence_id[23:16]<=byte_data;
                            5: sequence_id[31:24]<=byte_data;
                            6: length[7:0]<=byte_data;
                            7: begin
                                length[15:8]<=byte_data; position<=0;
                                if ({byte_data,length[7:0]}>2048) begin state<=0; errors<=errors+1'b1; end
                                else state<=({byte_data,length[7:0]}==0) ? 4 : 3;
                            end
                        endcase
                        header_index<=header_index+1'b1;
                    end
                    3: begin
                        payload[position]<=byte_data; crc<=crc16_byte(crc,byte_data);
                        if (position==length-1) state<=4; else position<=position+1'b1;
                    end
                    4: begin received_crc_low<=byte_data; state<=5; end
                    5: begin
                        state<=0;
                        if ({byte_data,received_crc_low}==crc) valid<=1;
                        else errors<=errors+1'b1;
                    end
                endcase
            end else if (state!=0) idle_count<=idle_count+1'b1;
        end
    end
endmodule
