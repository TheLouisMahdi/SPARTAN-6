`timescale 1ns/1ps
`include "suite.vh"
module tb_cnn;
    reg clk=0; always #10 clk=~clk;
    reg reset=1,start=0,model_we=0,input_we=0;
    reg [10:0] model_address=0,debug_address=0;
    reg [8:0] input_address=0;
    reg [7:0] model_data=0,input_data=0;
    wire b1,b2,d1,d2;
    wire signed [31:0] o1,c1,o2,c2;
    wire [31:0] t1,t2;
    wire [7:0] debug1,debug2;
    eye_cnn p1(clk,reset,start,model_we,model_address,model_data,input_we,input_address,input_data,
        b1,d1,o1,c1,t1,debug_address,debug1);
    eye_cnn_parallel p2(clk,reset,start,model_we,model_address,model_data,input_we,input_address,input_data,
        b2,d2,o2,c2,t2,debug_address,debug2);
    reg [7:0] inputs[0:`CASES*512-1],models[0:`CASES*1208-1];
    reg [7:0] pool1[0:`CASES*420-1],pool2[0:`CASES*96-1],fc1[0:`CASES*8-1];
    reg [31:0] logits[0:`CASES*2-1];
    integer n=-1,j,timeout,mac1,mac2,writes1,writes2;
    reg finished1=0,finished2=0,checking=0;
    task fail;
        input [255:0] reason;
        begin $display("FAIL case=%0d %0s",n,reason);$finish;end
    endtask
    task check_write;
        input [1:0] layer;
        input [8:0] address;
        input [7:0] value;
        begin
            case(layer)
                0:if(address>=420 || value!==pool1[n*420+address])fail("pool1 value/address");
                1:if(address>=96 || value!==pool2[n*96+address])fail("pool2 value/address");
                2:if(address>=8 || value!==fc1[n*8+address])fail("fc1 value/address");
            endcase
        end
    endtask
    always @(posedge clk)if(checking)begin
        if(p1.state==9)mac1=mac1+1;
        if(p2.state==9)begin
            mac2=mac2+1;
            if(p2.activation_address[0]!==p2.activation_address[1])fail("activation broadcast conflict");
            if(p2.pe[0].channel==p2.pe[1].channel)fail("duplicate output channels");
        end
        if(p1.state==12 && p1.layer<3)begin
            check_write(p1.layer,p1.output_index,p1.output_value);writes1=writes1+1;
        end
        if((p2.state==12 || p2.state==13) && p2.layer<3)begin
            check_write(p2.layer,p2.write_address,p2.write_value);writes2=writes2+1;
        end
    end
    initial begin
        $readmemh("inputs.hex",inputs);$readmemh("models.hex",models);
        $readmemh("pool1.hex",pool1);$readmemh("pool2.hex",pool2);
        $readmemh("fc1.hex",fc1);$readmemh("logits.hex",logits);
        repeat(3)@(negedge clk);reset=0;start=1;
        repeat(8)@(negedge clk);start=0;reset=1;
        repeat(3)@(negedge clk);if(b1 || b2 || d1 || d2)fail("reset while busy");reset=0;
        for(n=0;n<`CASES;n=n+1)begin
            for(j=0;j<1208;j=j+1)begin
                @(negedge clk);model_we=1;model_address=j;model_data=models[n*1208+j];
            end
            @(negedge clk);model_we=0;
            for(j=0;j<512;j=j+1)begin
                @(negedge clk);input_we=1;input_address=j;input_data=inputs[n*512+j];
            end
            @(negedge clk);input_we=0;debug_address=1207;
            repeat(3)@(negedge clk);
            if(debug1!==models[n*1208+1207] || debug2!==models[n*1208+1207])fail("model boundary readback");
            checking=1;mac1=0;mac2=0;writes1=0;writes2=0;finished1=0;finished2=0;start=1;
            @(negedge clk);start=0;timeout=0;
            while(!(finished1 && finished2) && timeout<500000)begin
                @(negedge clk);timeout=timeout+1;
                if(d1)begin
                    if(finished1 || b1)fail("P1 done pulse/busy");
                    if(o1!==logits[n*2] || c1!==logits[n*2+1])fail("P1 logits");finished1=1;
                end
                if(d2)begin
                    if(finished2 || b2)fail("P2 done pulse/busy");
                    if(o2!==logits[n*2] || c2!==logits[n*2+1])fail("P2 logits");finished2=1;
                end
            end
            if(!(finished1 && finished2))fail("timeout");
            if(mac1!=29728 || mac2!=14864)fail("useful concurrent MAC count");
            if(writes1!=524 || writes2!=524)fail("missing/extra activation writes");
            if(t2>=t1)fail("P2 no compute improvement");
            @(negedge clk);if(d1 || d2)fail("done wider than one cycle");checking=0;
            for(j=0;j<96;j=j+1)begin
                debug_address=1536+j;repeat(3)@(negedge clk);
                if(debug1!==pool2[n*96+j] || debug2!==pool2[n*96+j])fail("pool2 debug map");
            end
            $display("CASE_PASS %0d P1=%0d P2=%0d CONCURRENT=%0d",n,t1,t2,mac2);
        end
        $display("SUITE_PASS %0d synthetic cases, P1/P2 layers/logits/concurrency",`CASES);$finish;
    end
endmodule
