`timescale 1ns/1ps
// Full variant glue, actual pin-level UART and controller. CNN busy/claim are
// forced only to isolate the ownership policy; arithmetic is tested separately.
module BSCAN_SPARTAN6 #(parameter JTAG_CHAIN=1)(output CAPTURE,DRCK,RESET,RUNTEST,SEL,SHIFT,TCK,TDI,TMS,UPDATE,input TDO);
assign {CAPTURE,DRCK,RESET,RUNTEST,SEL,SHIFT,TCK,TDI,TMS,UPDATE}=0;endmodule
module BUFG(input I,output O);assign O=I;endmodule
module tb_jtag_ownership;
    reg clk=0,rx=1;always #10 clk=~clk;
    wire tx;wire[3:0]led;wire[7:0]seg;wire[5:0]digit;
    reg[7:0]bytes[0:29];integer i,j;
    eye_top board(clk,rx,tx,4'hf,led,seg,digit);
    task send_byte;
        input[7:0]value;
        begin
            @(negedge clk);rx=0;#1000;
            for(j=0;j<8;j=j+1)begin rx=value[j];#1000;end
            rx=1;#1000;
        end
    endtask
    initial begin
        $readmemh("uart_ownership.hex",bytes);
        #6000;force board.jtag_claim=1;force board.core_busy=1;
        repeat(10)@(posedge clk);
        for(i=0;i<15;i=i+1)send_byte(bytes[i]);
        wait(board.packet_valid);#100;
        if(board.uart_input_we||!board.jtag_lock)begin $display("FAIL active ownership");$finish;end
        release board.jtag_claim;
        repeat(20)begin
            @(posedge clk);#1;
            if(!board.jtag_lock||board.uart_input_we||!board.packet_valid)begin
                $display("FAIL TAP reset released active CNN ownership");$finish;
            end
        end
        release board.core_busy;
        wait(board.uart_upload_active);#100;
        if(board.pretrained_core.cnn.bank_a[0]!==-32'sd18504)begin $display("FAIL deferred UART write");$finish;end
        wait(!board.packet_valid&&!board.tx_packet_busy);#100;
        if(board.jtag_available)begin $display("FAIL partial UART upload not locked");$finish;end
        for(i=15;i<30;i=i+1)send_byte(bytes[i]);
        wait(board.commands.reply[0]==5);
        wait(!board.packet_valid&&!board.tx_packet_busy);#100;
        if(board.uart_upload_active||!board.jtag_available)begin $display("FAIL ownership recovery");$finish;end
        $display("JTAG_OWNERSHIP_PASS deferred_UART partial_upload TAP_reset_lock recovery");$finish;
    end
    initial begin #5000000;$display("FAIL ownership watchdog");$finish;end
endmodule
