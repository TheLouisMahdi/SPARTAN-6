`timescale 1ns/1ps
module BSCAN_SPARTAN6 #(parameter JTAG_CHAIN=1)(output reg CAPTURE=0,DRCK=0,RESET=1,RUNTEST=0,SEL=1,SHIFT=0,TCK=0,TDI=0,TMS=0,UPDATE=0,input TDO);endmodule
module BUFG(input I,output O);assign O=I;endmodule
module tb_tiny_candidate;
    reg clk=0,available=1;always #10 clk=~clk;
    wire busy,done,claim,we,start;wire[11:0]address;wire[7:0]data;
    wire signed[31:0]first,second,debug;wire[31:0]cycles;
    reg [7:0]vectors[0:39999];
    integer packet_offset=0,offset=0,size,status,seq,frame,eye,i,j,c,attempt,starts=0,writes=0,before_starts=0,before_writes=0;
    reg[63:0]capture_ns;reg[31:0]want_first,want_second;
    reg[327:0]answer;
    jtag_tiny_port dut(clk,available,1'b1,8'd1,32'h353bd6ca,busy,done,first,second,cycles,claim,we,address,data,start);
    wire [7:0] dbg;
    eye_cnn_parallel cnn(clk,dut.rrst,start,1'b0,11'd0,8'd0,we,address[8:0],data,busy,done,first,second,cycles,11'd0,dbg);
    always #42 dut.user1.TCK=~dut.user1.TCK;
    always @(posedge clk)begin
        if(start)starts=starts+1;
        if(we)begin
            if(address!==(writes-before_writes) || data!==(((vectors[packet_offset+67+(address/4)] >> ((3-address%4)*2)) & 3)*85))begin
                $display("FAIL symbol expansion address=%d data=%d",address,data);$finish;end
            writes=writes+1;
            if(busy||!claim||!available||address>511)begin $display("FAIL input ownership");$finish;end
        end
    end
    function[31:0]word;
        input integer a;
        begin word={vectors[a+3],vectors[a+2],vectors[a+1],vectors[a]};end
    endfunction
    task capture_scan;
        begin
            @(negedge dut.user1.TCK);dut.user1.CAPTURE=1;dut.user1.SHIFT=0;dut.user1.UPDATE=0;
            @(negedge dut.user1.TCK);dut.user1.CAPTURE=0;
        end
    endtask
    task bit_shift;
        input value;input integer bit_number;
        begin
            @(negedge dut.user1.TCK);dut.user1.SHIFT=1;dut.user1.TDI=value;#1;
            if(bit_number<328)answer[bit_number]=dut.tdo;
            @(posedge dut.user1.TCK);#1;
        end
    endtask
    task finish_scan;
        begin
            @(negedge dut.user1.TCK);dut.user1.SHIFT=0;
            @(negedge dut.user1.TCK);dut.user1.UPDATE=1;
            @(negedge dut.user1.TCK);dut.user1.UPDATE=0;
        end
    endtask
    initial begin
        $readmemh("cnn_vectors.hex",vectors,0,`VECTOR_BYTES-1);
        #101;dut.user1.RESET=0;repeat(8)@(negedge dut.user1.TCK);
        for(c=0;c<9;c=c+1)begin
            size={vectors[offset+1],vectors[offset]};status=vectors[offset+2];
            seq=word(offset+3);frame=word(offset+7);eye=vectors[offset+11];
            capture_ns={word(offset+16),word(offset+12)};want_first=word(offset+20);want_second=word(offset+24);
            offset=offset+28;packet_offset=offset;available=(status!=8);before_starts=starts;before_writes=writes;
            capture_scan;
            for(i=0;i<size;i=i+1)for(j=0;j<8;j=j+1)bit_shift(vectors[offset+i][j],i*8+j);
            finish_scan;offset=offset+size;
            attempt=0;answer=0;
            while(attempt<1000&&!(answer[31:0]==32'h3159544a&&answer[63:32]==seq&&answer[79:64]==146&&answer[103:96]==status))begin
                capture_scan;for(j=0;j<328;j=j+1)bit_shift(0,j);finish_scan;attempt=attempt+1;
            end
            if(attempt==1000)begin $display("FAIL CNN timeout/status case=%d reply=%h",c,answer);$finish;end
            if(status==0)begin
                $display("TINY_RESULT frame=%0d capture=%0d eye=%0d open=%0d closed=%0d cycles=%0d",frame,capture_ns,eye,$signed(answer[159:128]),$signed(answer[191:160]),answer[223:192]);
                if(answer[159:128]!==want_first||answer[191:160]!==want_second||answer[223:192]===0||
                   answer[255:224]!==frame||answer[319:256]!==capture_ns||answer[327:320]!==eye||starts!=before_starts+1||writes!=before_writes+512)begin
                    $display("FAIL logits/metadata/cycles case=%d reply=%h",c,answer);$finish;
                end
            end else if(starts!=before_starts)begin $display("FAIL invalid input started CNN");$finish;end
            if((status==8||status==9)&&writes!=before_writes)begin $display("FAIL busy/model input write");$finish;end
            $display("TINY_JTAG_CASE_PASS %d status=%d polls=%d",c,status,attempt);
        end
        $display("TINY_JTAG_SIM_PASS eyes=4 integer_mismatches=0 invalid_start=0 metadata=exact");$finish;
    end
    initial begin #100000000;$display("FAIL watchdog");$finish;end
endmodule
