// Included only when a Git baseline top is compiled alongside the current RTL.
wire baseline_tx;
wire [3:0] baseline_led;
wire [7:0] baseline_seg;
wire [5:0] baseline_digit;
eye_top_baseline #(.CLOCK_HZ(1000000),.BAUD(100000),.PE_COUNT(`SYSTEM_PE_COUNT),
    .PRETRAINED(`SYSTEM_PRETRAINED)) baseline(
    .clk(clk),.uart_rx_pin(rx),.uart_tx_pin(baseline_tx),.key(keys),
    .led(baseline_led),.seg(baseline_seg),.digit(baseline_digit));
integer equivalent_cycles=0;
always @(negedge clk)begin
    if({tx,led,seg,digit} !== {baseline_tx,baseline_led,baseline_seg,baseline_digit})
        fail("baseline UART/LED/display mismatch");
    if({dut.rx_accept,dut.read_address,dut.model_we,dut.model_address,dut.model_data,
        dut.input_we,dut.input_address,dut.input_data,dut.core_start,dut.core_busy,dut.core_done,
        dut.core_open,dut.core_closed,dut.core_cycles,dut.debug_address,dut.debug_data,
        dut.tx_start,dut.reply_length,dut.page,dut.result_valid,dut.result_flags,dut.frames,
        dut.last_pair_cycles,dut.capture_ms,dut.timing_mode,dut.timing_valid,dut.left_score,dut.right_score}
        !==
       {baseline.rx_accept,baseline.read_address,baseline.model_we,baseline.model_address,baseline.model_data,
        baseline.input_we,baseline.input_address,baseline.input_data,baseline.core_start,baseline.core_busy,baseline.core_done,
        baseline.core_open,baseline.core_closed,baseline.core_cycles,baseline.debug_address,baseline.debug_data,
        baseline.tx_start,baseline.reply_length,baseline.page,baseline.result_valid,baseline.result_flags,baseline.frames,
        baseline.last_pair_cycles,baseline.capture_ms,baseline.timing_mode,baseline.timing_valid,baseline.left_score,baseline.right_score})
        fail("baseline control/memory/handshake mismatch");
    if({dut.commands.state,dut.commands.command,dut.commands.sequence_id,dut.commands.position,
        dut.commands.uploaded,dut.commands.model_loaded,dut.commands.pair_elapsed}
        !== {baseline.state,baseline.command,baseline.sequence_id,baseline.position,
             baseline.uploaded,baseline.model_loaded,baseline.pair_elapsed})
        fail("baseline command state mismatch");
    equivalent_cycles=equivalent_cycles+1;
end
