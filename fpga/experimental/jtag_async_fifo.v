// 16x9 dual-clock distributed FIFO. Gray pointers cross through two registers.
// Reset asserts in both domains; the wrapper synchronizes its release separately.
module jtag_async_fifo(input wclk,input wrst,input wen,input [8:0] wdata,
    output reg full=0,output reg overflow=0,
    input rclk,input rrst,input ren,output wire [8:0] rdata,output reg empty=1);
    (* ram_style="distributed" *) reg [8:0] mem[0:15];
    reg [4:0] wb=0,wg=0,rb=0,rg=0;
    (* ASYNC_REG="TRUE", SHREG_EXTRACT="NO" *) reg [4:0] rmeta=0,rsync=0,wmeta=0,wsync=0;
    wire [4:0] wn=wb+(wen&&!full),rn=rb+(ren&&!empty);
    wire [4:0] wgn=(wn>>1)^wn,rgn=(rn>>1)^rn;
    assign rdata=mem[rb[3:0]];
    always @(posedge wclk or posedge wrst)begin
        if(wrst)begin wb<=0;wg<=0;rmeta<=0;rsync<=0;full<=0;overflow<=0;end
        else begin
            rmeta<=rg;rsync<=rmeta;
            if(wen&&!full)mem[wb[3:0]]<=wdata;
            if(wen&&full)overflow<=1;
            wb<=wn;wg<=wgn;full<=(wgn=={~rsync[4:3],rsync[2:0]});
        end
    end
    always @(posedge rclk or posedge rrst)begin
        if(rrst)begin rb<=0;rg<=0;wmeta<=0;wsync<=0;empty<=1;end
        else begin
            wmeta<=wg;wsync<=wmeta;
            rb<=rn;rg<=rgn;empty<=(rgn==wsync);
        end
    end
endmodule
