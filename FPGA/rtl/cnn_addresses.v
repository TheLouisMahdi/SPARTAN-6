// Explicit constant shift/add addressing; reserve DSP arithmetic for the MAC.
(* use_dsp48="no" *) module cnn_addresses(input wire [1:0] layer,
    input wire [3:0] out_channel,input wire [3:0] out_x,input wire [2:0] out_y,
    input wire [2:0] in_channel,input wire [1:0] kernel_x,input wire [1:0] kernel_y,
    input wire [1:0] pool_cell,input wire [6:0] feature,
    output reg [10:0] weight_address,output reg [8:0] activation_address,
    output reg [4:0] bias_base);
    wire [10:0] oc={7'd0,out_channel},ic={8'd0,in_channel};
    wire [10:0] kx={9'd0,kernel_x},ky={9'd0,kernel_y};
    wire [10:0] row=({8'd0,out_y}<<1)+{10'd0,pool_cell[1]}+ky;
    wire [10:0] column=({7'd0,out_x}<<1)+{10'd0,pool_cell[0]}+kx;
    wire [10:0] index={4'd0,feature};
    always @* begin
        case(layer)
            0:begin
                weight_address=(oc<<3)+oc+(ky<<1)+ky+kx;
                activation_address=(row<<5)+column;
                bias_base=0;
            end
            1:begin
                weight_address=11'd36+(oc<<5)+(oc<<2)+(ic<<3)+ic+(ky<<1)+ky+kx;
                activation_address=(ic<<6)+(ic<<5)+(ic<<3)+ic+(row<<4)-row+column;
                bias_base=4;
            end
            2:begin
                weight_address=11'd324+(oc<<6)+(oc<<5)+index;
                activation_address=feature;bias_base=12;
            end
            default:begin
                weight_address=11'd1092+(oc<<3)+index;
                activation_address=feature;bias_base=20;
            end
        endcase
    end
endmodule
