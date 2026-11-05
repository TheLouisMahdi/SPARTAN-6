`timescale 1ns/1ps
`include "suite.vh"
module tb_omz;
    reg clk=0,reset=1,start=0,input_we=0;
    always #5 clk=~clk;
    reg [11:0] input_address=0;
    reg [7:0] input_data=0;
    wire busy,done;
    wire signed [31:0] opened,closed,debug_data;
    wire [31:0] cycles;
    omz_cnn #(`OMZ_PE_COUNT) dut(clk,reset,start,input_we,input_address,input_data,
        busy,done,opened,closed,cycles,12'd0,debug_data);
    reg [7:0] inputs[0:`CASES*3072-1];
    reg signed [31:0] expected[0:`CASES*3772-1];
    integer test_case=0,i,timeout,checked=0,base,commits=0;
    always @(posedge clk)begin
        if(!reset && dut.busy && dut.write_output && dut.layer<3)begin
            base=test_case*3772+(dut.layer==0 ? 0 : dut.layer==1 ? 2250 : 2970);
            if(dut.write_value !== expected[base+dut.write_address])begin
                $display("OMZ_FAIL case=%0d layer=%0d address=%0d got=%0d expected=%0d",
                    test_case,dut.layer,dut.write_address,dut.write_value,expected[base+dut.write_address]);
                $finish;
            end
            checked=checked+1;
        end
        if(!reset && dut.state==dut.DOT_REQ && dut.coefficient>=2 && !dut.product_valid)begin
            $display("OMZ_FAIL bubble in steady-state MAC pipeline");$finish;
        end
        if(!reset && dut.product_valid)commits=commits+1;
    end
    initial begin
        $readmemh("inputs.hex",inputs);$readmemh("expected.hex",expected);
        repeat(4)@(negedge clk);reset=0;
        // Abort a filled pipeline, then start a complete inference from new input.
        start=1;@(negedge clk);start=0;
        wait(dut.product_valid);@(negedge clk);reset=1;
        repeat(2)@(negedge clk);
        if(busy || done || dut.read_valid || dut.product_valid)begin
            $display("OMZ_FAIL reset did not flush pipeline");$finish;
        end
        reset=0;commits=0;
        for(test_case=0;test_case<`CASES;test_case=test_case+1)begin
            for(i=0;i<3072;i=i+1)begin
                input_we=1;input_address=i;input_data=inputs[test_case*3072+i];@(negedge clk);
            end
            input_we=0;start=1;@(negedge clk);start=0;timeout=0;
            while(!done && timeout<3000000)begin @(negedge clk);timeout=timeout+1;end
            if(!done || opened !== expected[test_case*3772+3770] || closed !== expected[test_case*3772+3771])begin
                $display("OMZ_FAIL logits case=%0d got=%0d,%0d expected=%0d,%0d timeout=%0d",
                    test_case,opened,closed,expected[test_case*3772+3770],expected[test_case*3772+3771],timeout);$finish;
            end
            $display("OMZ_CASE_PASS %0d cycles=%0d",test_case,cycles);
            @(negedge clk);
        end
        if(checked!=`CASES*3770)begin $display("OMZ_FAIL write coverage %0d",checked);$finish;end
        if(commits!=`CASES*(647800/`OMZ_PE_COUNT))begin
            $display("OMZ_FAIL MAC coverage %0d",commits);$finish;
        end
        $display("OMZ_PASS cases=%0d writes=%0d MAC_commits=%0d",`CASES,checked,commits);$finish;
    end
endmodule
