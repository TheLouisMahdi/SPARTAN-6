module eye_top #(parameter integer CLOCK_HZ=50000000, parameter integer BAUD=1000000,
    parameter integer PE_COUNT=2, parameter integer PRETRAINED=1)(
    input wire clk, input wire uart_rx_pin, output wire uart_tx_pin,
    input wire [3:0] key, output wire [3:0] led,
    output wire [7:0] seg, output wire [5:0] digit);
    reg [7:0] startup=8'hff;
    wire reset=(startup!=0);
    always @(posedge clk) if(startup!=0)startup<=startup-1'b1;
    wire [7:0] rx_byte,tx_byte,packet_kind,read_data,reply_kind,reply_data;
    wire rx_byte_valid,rx_framing,rx_accept,packet_valid;
    wire tx_start,tx_packet_done,tx_packet_busy,tx_byte_start,tx_byte_busy;
    wire [31:0] packet_sequence,reply_sequence;
    wire [15:0] packet_length;
    wire [10:0] read_address,model_address,debug_address;
    wire [5:0] reply_length,reply_address;
    wire model_we,input_we,core_start,core_busy,core_done;
    wire [11:0] input_address;
    wire [7:0] model_data,input_data,debug_data;
    wire signed [31:0] core_open,core_closed;
    wire [31:0] core_cycles,capture_ms;
    wire [1:0] page,timing_valid;
    wire result_valid,timing_mode;
    wire [3:0] result_flags,decisions;
    wire [23:0] frames,last_pair_cycles;
    wire [15:0] left_score,right_score;
    wire clear_blinks,invalidate_blinks,start_pair,pair_active,result_update,timed_result,timing_update,timing_reply;
    wire [3:0] key_pressed,key_pulse;
    wire [1:0] activity_pulse;
    wire [7:0] fps_bcd;
    wire [11:0] milliseconds_bcd;
    wire [1:0] blink_events;
    wire [31:0] left_blinks,right_blinks;
    wire [1:0] timing_closed,timing_known;
    wire timing_event;
    wire [31:0] timing_count,timing_long,left_current_ms,right_current_ms;
    wire [31:0] last_left_ms,last_right_ms,last_overlap_ms,last_duration_ms;
    wire [31:0] current_ms=left_current_ms>right_current_ms ? left_current_ms : right_current_ms;
    wire [31:0] displayed_ms=current_ms!=0 ? current_ms : last_duration_ms;
    wire [9:0] capped_ms=displayed_ms>999 ? 10'd999 : displayed_ms[9:0];
    function [11:0] decimal3;
        input [9:0] value;
        reg [11:0] bcd;
        integer bit_index,digit_index;
        begin
            bcd=0;
            for(bit_index=9;bit_index>=0;bit_index=bit_index-1)begin
                for(digit_index=0;digit_index<3;digit_index=digit_index+1)
                    if(bcd[digit_index*4 +: 4]>=5)bcd[digit_index*4 +: 4]=bcd[digit_index*4 +: 4]+4'd3;
                bcd={bcd[10:0],value[bit_index]};
            end
            decimal3=bcd;
        end
    endfunction
    uart_rx #((CLOCK_HZ+BAUD/2)/BAUD) receiver(clk,reset,uart_rx_pin,rx_byte,rx_byte_valid,rx_framing);
    uart_tx #((CLOCK_HZ+BAUD/2)/BAUD) transmitter(clk,reset,tx_byte_start,tx_byte,uart_tx_pin,tx_byte_busy);
    packet_rx #(CLOCK_HZ) packets_in(.clk(clk),.reset(reset),.byte_data(rx_byte),.byte_valid(rx_byte_valid),
        .framing_error(rx_framing),.accept(rx_accept),.valid(packet_valid),.kind(packet_kind),
        .sequence_id(packet_sequence),.length(packet_length),.read_address(read_address),.read_data(read_data),.errors());
    packet_tx packets_out(clk,reset,tx_start,reply_kind,reply_sequence,reply_length,
        reply_address,reply_data,tx_byte_start,tx_byte,tx_byte_busy,tx_packet_busy,tx_packet_done);
    command_controller #(.PRETRAINED(PRETRAINED)) commands(
        .clk(clk), .reset(reset), .packet_valid(packet_valid),
        .packet_kind(packet_kind), .packet_sequence(packet_sequence), .packet_length(packet_length),
        .rx_accept(rx_accept), .read_address(read_address), .read_data(read_data),
        .tx_start(tx_start), .tx_packet_done(tx_packet_done), .reply_kind(reply_kind),
        .reply_sequence(reply_sequence), .reply_length(reply_length), .reply_address(reply_address),
        .reply_data(reply_data), .model_we(model_we), .model_address(model_address),
        .model_data(model_data), .input_we(input_we), .input_address(input_address),
        .input_data(input_data), .core_start(core_start), .core_done(core_done),
        .core_open(core_open), .core_closed(core_closed), .core_cycles(core_cycles),
        .debug_address(debug_address), .debug_data(debug_data), .key_pulse(key_pulse),
        .page(page), .result_valid(result_valid), .result_flags(result_flags),
        .frames(frames), .last_pair_cycles(last_pair_cycles), .decisions(decisions),
        .timing_mode(timing_mode), .capture_ms(capture_ms), .timing_valid(timing_valid),
        .left_score(left_score), .right_score(right_score), .clear_blinks(clear_blinks),
        .invalidate_blinks(invalidate_blinks), .start_pair(start_pair), .pair_active(pair_active),
        .result_update(result_update), .timed_result(timed_result), .timing_update(timing_update),
        .timing_reply(timing_reply), .blink_events(blink_events), .left_blinks(left_blinks),
        .right_blinks(right_blinks), .timing_closed(timing_closed), .timing_known(timing_known),
        .timing_event(timing_event), .timing_count(timing_count), .timing_long(timing_long),
        .left_current_ms(left_current_ms), .right_current_ms(right_current_ms), .last_left_ms(last_left_ms),
        .last_right_ms(last_right_ms), .last_overlap_ms(last_overlap_ms), .last_duration_ms(last_duration_ms));
    generate if(PRETRAINED)begin: pretrained_core
        wire signed [31:0] unused_debug;
        omz_cnn #(PE_COUNT) cnn(clk,reset,core_start,input_we,input_address,input_data,
            core_busy,core_done,core_open,core_closed,core_cycles,12'd0,unused_debug);
        assign debug_data=0;
    end else if(PE_COUNT==1)begin: serial_core
        eye_cnn cnn(clk,reset,core_start,model_we,model_address,model_data,input_we,input_address[8:0],input_data,
            core_busy,core_done,core_open,core_closed,core_cycles,debug_address,debug_data);
    end else if(PE_COUNT==2)begin: parallel_core
        eye_cnn_parallel cnn(clk,reset,core_start,model_we,model_address,model_data,input_we,input_address[8:0],input_data,
            core_busy,core_done,core_open,core_closed,core_cycles,debug_address,debug_data);
    end else begin: invalid_pe_count
        PE_COUNT_must_be_1_or_2 invalid_configuration();
    end endgenerate
    key_filter #(CLOCK_HZ) buttons(clk,key,key_pressed,key_pulse);
    activity_status #(CLOCK_HZ) activity(clk,reset,result_update || timing_reply,result_valid,
        timing_mode ? timing_closed : decisions[3:2],activity_pulse,fps_bcd);
    blink_event left_events(clk,clear_blinks,invalidate_blinks,result_update,timed_result,
        decisions[2],capture_ms,blink_events[0],left_blinks);
    blink_event right_events(clk,clear_blinks,invalidate_blinks,result_update,timed_result,
        decisions[3],capture_ms,blink_events[1],right_blinks);
    binocular_timer timing(clk,clear_blinks,invalidate_blinks,timing_update,capture_ms,timing_valid,
        left_score,right_score,timing_closed,timing_known,timing_event,timing_count,timing_long,
        left_current_ms,right_current_ms,last_left_ms,last_right_ms,last_overlap_ms,last_duration_ms);
    latency_meter #(CLOCK_HZ) latency(clk,reset,start_pair,
        pair_active,result_update,milliseconds_bcd);
    board_display #(CLOCK_HZ) display(clk,page,key_pressed[3],result_valid,result_flags,frames,last_pair_cycles,
        timing_mode ? decimal3(capped_ms) : milliseconds_bcd,seg,digit);
    assign led=key_pressed[3] ? 4'hf :
        {result_valid&&result_flags[3],activity_pulse[1],activity_pulse[0],result_valid&&result_flags[2]};
endmodule
