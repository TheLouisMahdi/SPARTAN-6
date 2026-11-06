`timescale 1ns/1ps
// Pin-level model upload is deferred while JTAG owns the common core/input path.
module BSCAN_SPARTAN6 #(parameter JTAG_CHAIN=1)(output CAPTURE,DRCK,RESET,RUNTEST,SEL,SHIFT,TCK,TDI,TMS,UPDATE,input TDO);
assign {CAPTURE,DRCK,RESET,RUNTEST,SEL,SHIFT,TCK,TDI,TMS,UPDATE}=0;endmodule
module BUFG(input I,output O);assign O=I;endmodule
module tb_tiny_ownership;
    reg clk=0,rx=1;always #10 clk=~clk;
    wire tx;wire[3:0]led;wire[7:0]seg;wire[5:0]digit;
    reg[7:0]bytes[0:`MODEL_PACKET_BYTES-1];integer i,j,writes=0;
    eye_top board(clk,rx,tx,4'hf,led,seg,digit);
    always @(posedge clk)if(board.model_we)begin
        writes=writes+1;
        if(board.jtag_lock||board.jtag_claim)begin $display("FAIL model write overlaps JTAG");$finish;end
    end
    task send_byte;
        input[7:0]value;
        begin
            @(negedge clk);rx=0;#1000;
            for(j=0;j<8;j=j+1)begin rx=value[j];#1000;end
            rx=1;#1000;
        end
    endtask
    initial begin
        $readmemh("model_upload.hex",bytes);
        #6000;force board.jtag_claim=1;force board.core_busy=1;
        repeat(10)@(posedge clk);
        for(i=0;i<`MODEL_PACKET_BYTES;i=i+1)send_byte(bytes[i]);
        wait(board.packet_valid);#100;
        if(board.model_we||!board.jtag_lock)begin $display("FAIL active ownership");$finish;end
        release board.jtag_claim;
        repeat(20)begin
            @(posedge clk);#1;
            if(!board.jtag_lock||board.model_we||!board.packet_valid)begin
                $display("FAIL TAP reset released busy ownership");$finish;
            end
        end
        release board.core_busy;
        wait(board.model_we);
        while(board.packet_valid)begin
            @(posedge clk);#1;
            if(board.packet_valid&&board.jtag_available)begin $display("FAIL UART upload ownership");$finish;end
        end
        wait(!board.tx_packet_busy);repeat(5)@(posedge clk);#1;
        if(writes!=1208||!board.transport_model_loaded||board.transport_method!=1||!board.jtag_available)begin
            $display("FAIL model upload/recovery writes=%d",writes);$finish;
        end
        $display("TINY_OWNERSHIP_PASS deferred_UART model_upload TAP_reset_lock recovery");$finish;
    end
    initial begin #20000000;$display("FAIL ownership watchdog");$finish;end
endmodule
