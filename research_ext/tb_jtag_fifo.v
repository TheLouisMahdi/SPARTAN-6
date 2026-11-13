`timescale 1ns/1ps
module tb_jtag_fifo;
    reg wc=0,rc=0,wr=1,rr=1,we=0,re=0;
    always #3 wc=~wc;always #17 rc=~rc;
    reg[8:0]data=0;wire[8:0]out;wire full,empty,overflow;
    integer i;
    jtag_async_fifo dut(wc,wr,we,data,full,overflow,rc,rr,re,out,empty);
    initial begin
        #101;wr=0;rr=0;
        for(i=0;i<16;i=i+1)begin @(negedge wc);we=1;data=i;end
        @(negedge wc);we=0;
        if(!full)begin $display("FAIL full");$finish;end
        @(negedge wc);we=1;data=99;@(negedge wc);we=0;
        if(!overflow)begin $display("FAIL overflow");$finish;end
        for(i=0;i<16;i=i+1)begin
            @(negedge rc);re=1;
            if(empty||out!==i)begin $display("FAIL FIFO order %d got %d",i,out);$finish;end
        end
        @(negedge rc);re=0;
        if(!empty)begin $display("FAIL empty");$finish;end
        wr=1;rr=1;#101;wr=0;rr=0;#200;
        if(full||overflow||!empty)begin $display("FAIL reset");$finish;end
        $display("JTAG_FIFO_PASS 16 ordered bytes overflow reset asynchronous clocks");$finish;
    end
    initial begin #100000;$display("FAIL watchdog");$finish;end
endmodule
