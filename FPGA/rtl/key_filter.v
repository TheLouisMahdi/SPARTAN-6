module key_filter #(parameter integer CLOCK_HZ=50000000)(input wire clk,
    input wire [3:0] key, output reg [3:0] pressed=0, output reg [3:0] pulse=0);
    (* ASYNC_REG="TRUE" *) reg [3:0] meta=0,synced=0;
    reg [19:0] count[0:3];
    integer k;
    initial for(k=0;k<4;k=k+1) count[k]=0;
    always @(posedge clk) begin
        meta<=~key; synced<=meta; pulse<=0;
        for(k=0;k<4;k=k+1) begin
            if(synced[k]==pressed[k]) count[k]<=0;
            else if(count[k]==CLOCK_HZ/50-1) begin
                count[k]<=0; pressed[k]<=synced[k]; pulse[k]<=synced[k];
            end else count[k]<=count[k]+1'b1;
        end
    end
endmodule
