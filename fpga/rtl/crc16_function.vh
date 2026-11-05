function [15:0] crc16_byte;
    input [15:0] previous;
    input [7:0] value;
    reg [15:0] c;
    integer bit_number;
    begin
        c=previous ^ {value,8'd0};
        for (bit_number=0;bit_number<8;bit_number=bit_number+1)
            c=c[15] ? (c<<1)^16'h1021 : c<<1;
        crc16_byte=c;
    end
endfunction
