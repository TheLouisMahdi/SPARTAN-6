`timescale 1ns/1ps
`ifndef TCK_HALF
`define TCK_HALF 42
`endif
module tb_jtag_probe;
    reg clk=0,tck=0,run_tck=1,run_sys=1;
    always #10 if(run_sys)clk=~clk;
    always #`TCK_HALF if(run_tck)tck=~tck;
    reg reset=1,sel=1,capture=0,shift=0,update=0,tdi=0;
    wire tdo;wire[3:0]led;
    jtag_probe_logic dut(clk,tck,reset,sel,capture,shift,update,tdi,tdo,led);
    reg[7:0]vectors[0:20000];
    reg[127:0]answer;
    integer offset=0,i,j,size,expected_status,expected_seq,case_id,echo_errors=0;
    reg last_bit;
    task start_scan;
        begin
            @(negedge tck);capture=1;shift=0;update=0;
            @(negedge tck);capture=0;
        end
    endtask
    task send_bit;
        input value;
        input integer bit_position;
        input check_echo;
        begin
            @(negedge tck);shift=1;tdi=value;#1;
            if(check_echo&&bit_position>=128&&tdo!==last_bit)echo_errors=echo_errors+1;
            if(bit_position<128)answer[bit_position]=tdo;
            @(posedge tck);#1;last_bit=value;
        end
    endtask
    task end_scan;
        begin
            @(negedge tck);shift=0;
            @(negedge tck);update=1;
            @(negedge tck);update=0;
        end
    endtask
    task read_reply;
        integer retry;
        begin
            answer=128'h0000010800000000000000003147544a;retry=0;
            while(answer[103:96]==8&&retry<4)begin
                #500;start_scan;
                for(j=0;j<128;j=j+1)send_bit(0,j,0);
                end_scan;retry=retry+1;
            end
            if(answer[31:0]!==32'h3147544a||answer[111:104]!==1)begin
                $display("FAIL signature %h",answer);$finish;
            end
        end
    endtask
    initial begin
        $readmemh("probe_vectors.hex",vectors,0,`VECTOR_BYTES-1);
        #101;reset=0;
        repeat(8)@(negedge tck);
        for(case_id=0;case_id<15;case_id=case_id+1)begin
            size=vectors[offset]+256*vectors[offset+1];
            expected_status=vectors[offset+2];
            expected_seq=vectors[offset+3]+256*vectors[offset+4]+65536*vectors[offset+5]+16777216*vectors[offset+6];
            offset=offset+7;
            start_scan;last_bit=0;
            for(i=0;i<size;i=i+1)begin
                for(j=0;j<8;j=j+1)send_bit(vectors[offset+i][j],i*8+j,1);
                if(i==32)begin
                    @(negedge tck);shift=0;run_tck=0;#1007;run_tck=1;
                end
            end
            if(case_id==9)for(j=0;j<3;j=j+1)send_bit(0,size*8+j,1);
            end_scan;offset=offset+size;
            read_reply;
            if(answer[103:96]!==expected_status||answer[63:32]!==expected_seq)begin
                $display("FAIL case=%d status=%d expected=%d seq=%h expected=%h",case_id,answer[103:96],expected_status,answer[63:32],expected_seq);$finish;
            end
            $display("JTAG_CASE_PASS %d status=%d",case_id,expected_status);
        end
        if(echo_errors!=0)begin $display("FAIL echo errors=%d",echo_errors);$finish;end
        // No writes when USER chain is deselected; reset clears both FIFO domains.
        sel=0;repeat(30)@(negedge tck);sel=1;
        reset=1;#131;reset=0;repeat(8)@(negedge tck);
        read_reply;
        if(answer[103:96]!==9)begin $display("FAIL reset");$finish;end
        $display("JTAG_PROBE_PASS HALF=%d cases=15 paused_TCK echo CDC reset",`TCK_HALF);$finish;
    end
    initial begin #200000000;$display("FAIL watchdog");$finish;end
endmodule
