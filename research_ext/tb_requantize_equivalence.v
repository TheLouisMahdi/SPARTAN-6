`timescale 1ns/1ps
module tb_requantize_equivalence;
    reg clk=0;always #10 clk=~clk;
    reg reset=1,start=0;reg signed[31:0]accumulator=0;reg[31:0]multiplier=0;
    wire a_busy,b_busy,a_done,b_done;wire[7:0]a_value,b_value;
    reg[71:0]vectors[0:`RQ_CASES-1];integer i,j;
    requantize_q24 a(clk,reset,start,accumulator,multiplier,a_busy,a_done,a_value);
    requantize_q24_dsp b(clk,reset,start,accumulator,multiplier,b_busy,b_done,b_value);
    always @(posedge clk)begin
        #1;
        if(a_busy!==b_busy||a_done!==b_done||(a_done&&a_value!==b_value))begin
            $display("FAIL requantizer schedule/value equivalence");$finish;end
    end
    initial begin
        $readmemh("rq_vectors.hex",vectors);
        repeat(3)@(negedge clk);reset=0;
        for(i=0;i<`RQ_CASES;i=i+1)begin
            @(negedge clk);accumulator=vectors[i][71:40];multiplier=vectors[i][39:8];start=1;
            @(negedge clk);start=0;
            j=0;
            while(!b_done&&j<40)begin @(negedge clk);j=j+1;end
            if(!b_done||b_value!==vectors[i][7:0])begin
                $display("FAIL Q24 golden case=%d result=%d expected=%d",i,b_value,vectors[i][7:0]);$finish;end
        end
        $display("REQUANTIZE_EQUIVALENCE_PASS cases=%0d",`RQ_CASES);$finish;
    end
endmodule
