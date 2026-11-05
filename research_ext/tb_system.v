`timescale 1ns/1ps
`include "system.vh"
// Actual serial pins, framing, CRC, RAMs, CNN, temporal logic and board outputs.
module tb_system;
    reg clk=0,rx=1;
    reg [3:0] keys=4'hf;
    always #5 clk=~clk;
    wire tx;
    wire [3:0] led;
    wire [7:0] seg;
    wire [5:0] digit;
    eye_top #(.CLOCK_HZ(1000000),.BAUD(100000),.PE_COUNT(`SYSTEM_PE_COUNT),.PRETRAINED(`SYSTEM_PRETRAINED)) dut(
        .clk(clk),.uart_rx_pin(rx),.uart_tx_pin(tx),.key(keys),.led(led),.seg(seg),.digit(digit));
    `ifdef CHECK_BASELINE
        `include "top_equivalence.vh"
    `endif
    wire [7:0] received_byte;
    wire received_valid,framing_error;
    uart_rx #(10) monitor(clk,dut.reset,tx,received_byte,received_valid,framing_error);
    reg [7:0] requests[0:`REQUEST_BYTES-1];
    reg [15:0] sizes[0:`REQUESTS-1];
    reg [7:0] reply_expected[0:`REQUESTS-1],led_expected[0:`REQUESTS-1];
    reg [7:0] response[0:127];
    reg [5:0] seen_digits=0;
    integer received=0,target=65535,log_file,index,offset=0,j,k,timeout;
    task fail;
        input [511:0] message;
        begin $display("SYSTEM_FAIL request=%0d %0s",index,message);$finish;end
    endtask
    always @(posedge clk) begin
        if(!dut.reset)begin
            seen_digits<=seen_digits | ~digit;
            if((^seg)===1'bx || (^digit)===1'bx)fail("unknown display output");
            if(!dut.commands.model_loaded && dut.core_start)fail("CNN started without a model");
        end
        if(framing_error)fail("reply UART framing");
        if(received_valid)begin
            if(received>=128)fail("oversized response");
            response[received]=received_byte;
            $fdisplay(log_file,"%02x",received_byte);
            if(received==9)target=12+{received_byte,response[8]};
            received=received+1;
        end
    end
    task send_byte;
        input [7:0] value;
        integer bit_index;
        begin
            rx=0;repeat(10)@(negedge clk);
            for(bit_index=0;bit_index<8;bit_index=bit_index+1)begin
                rx=value[bit_index];repeat(10)@(negedge clk);
            end
            rx=1;repeat(10)@(negedge clk);
        end
    endtask
    initial begin
        $readmemh("requests.hex",requests);$readmemh("request_sizes.hex",sizes);
        $readmemh("reply_expected.hex",reply_expected);$readmemh("led_expected.hex",led_expected);
        log_file=$fopen("replies.hex","w");
        repeat(300)@(negedge clk);
        for(index=0;index<`REQUESTS;index=index+1)begin
            received=0;target=65535;
            for(j=0;j<sizes[index];j=j+1)send_byte(requests[offset+j]);
            offset=offset+sizes[index];timeout=0;
            if(reply_expected[index]!=0)begin
                while((received<12 || received<target) && timeout<600000)begin
                    @(negedge clk);timeout=timeout+1;
                end
                if(received!=target)fail("reply timeout/length");
            end else begin
                repeat(500)@(negedge clk);
                if(received!=0)fail("corrupted request received a reply");
            end
            repeat(40)@(negedge clk);
            if(led_expected[index]!=8'hff && (led & 4'h9)!=led_expected[index][3:0])fail("eye LED mapping");
            $display("SYSTEM_REQUEST_PASS %0d bytes=%0d",index,received);
        end
        if(seen_digits!=6'h3f)fail("not all six digits scanned");
        `ifdef CHECK_BASELINE
            // Debounced page selection, frame clear and lamp test after packet checks.
            for(k=0;k<4;k=k+1)begin
                keys=~(4'b0001<<k);repeat(20010)@(negedge clk);
                keys=4'hf;repeat(20010)@(negedge clk);
            end
            $display("EQUIVALENCE_PASS cycles=%0d",equivalent_cycles);
        `endif
        $fclose(log_file);$display("SYSTEM_PASS requests=%0d",`REQUESTS);$finish;
    end
endmodule
