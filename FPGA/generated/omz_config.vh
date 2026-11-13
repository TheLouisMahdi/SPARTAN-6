localparam [31:0] OMZ_FINGERPRINT=32'h4daa1000;
function [5:0] omz_shift;
input [1:0] layer;
begin
case(layer)
2'd0:omz_shift=6'd14;
2'd1:omz_shift=6'd14;
2'd2:omz_shift=6'd15;
2'd3:omz_shift=6'd15;
endcase
end
endfunction
