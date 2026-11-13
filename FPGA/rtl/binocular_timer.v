// Capture timestamps, not UART arrival times. Coefficients are unsigned Q16.
// Counts completed overlapping closures; long closures are retained separately.
module binocular_timer #(parameter integer MINIMUM_MS=30,LONG_MS=600,
    SYNC_MS=150,MAXIMUM_GAP_MS=250)(
    input wire clk,input wire reset,input wire invalidate,input wire update,
    input wire [31:0] timestamp_ms,input wire [1:0] valid_mask,
    input wire [15:0] left_score,input wire [15:0] right_score,
    output reg [1:0] closed_mask=0,output reg [1:0] known_mask=0,
    output reg event_pulse=0,output reg [31:0] count=0,long_count=0,
    output reg [31:0] left_current_ms=0,right_current_ms=0,
    output reg [31:0] last_left_ms=0,last_right_ms=0,last_overlap_ms=0,last_duration_ms=0);
    reg have_previous=0,armed=0,active=0,overlap_seen=0;
    reg [1:0] seen=0;
    reg [31:0] previous_ms=0,left_start=0,right_start=0,left_end=0,right_end=0,union_start=0;
    wire [31:0] gap=timestamp_ms-previous_ms;
    wire continuous=have_previous && gap!=0 && gap<=MAXIMUM_GAP_MS;
    wire [1:0] next_known={valid_mask[1] && (right_score<=26214 || right_score>=39321 || (continuous && known_mask[1])),
                           valid_mask[0] && (left_score<=26214 || left_score>=39321 || (continuous && known_mask[0]))};
    wire [1:0] next_closed={valid_mask[1] && (right_score>=39321 || (right_score>26214 && continuous && closed_mask[1])),
                            valid_mask[0] && (left_score>=39321 || (left_score>26214 && continuous && closed_mask[0]))};
    wire [31:0] start_l=next_closed[0] && (!continuous || !known_mask[0] || !closed_mask[0]) ? timestamp_ms : left_start;
    wire [31:0] start_r=next_closed[1] && (!continuous || !known_mask[1] || !closed_mask[1]) ? timestamp_ms : right_start;
    wire [31:0] end_l=closed_mask[0] && !next_closed[0] ? timestamp_ms : left_end;
    wire [31:0] end_r=closed_mask[1] && !next_closed[1] ? timestamp_ms : right_end;
    wire [31:0] duration_l=end_l-start_l,duration_r=end_r-start_r;
    wire [31:0] onset_lr=start_l-start_r,onset_rl=start_r-start_l;
    wire [31:0] relative_start_l=start_l-union_start,relative_start_r=start_r-union_start;
    wire [31:0] relative_end_l=end_l-union_start,relative_end_r=end_r-union_start;
    wire [31:0] overlap_start=relative_start_l>relative_start_r ? relative_start_l : relative_start_r;
    wire [31:0] overlap_end=relative_end_l<relative_end_r ? relative_end_l : relative_end_r;
    wire [31:0] overlap_ms=overlap_end-overlap_start;
    always @(posedge clk) begin
        event_pulse<=0;
        if(reset)begin
            have_previous<=0;armed<=0;active<=0;overlap_seen<=0;seen<=0;
            closed_mask<=0;known_mask<=0;count<=0;long_count<=0;
            previous_ms<=0;left_start<=0;right_start<=0;left_end<=0;right_end<=0;union_start<=0;
            left_current_ms<=0;right_current_ms<=0;
            last_left_ms<=0;last_right_ms<=0;last_overlap_ms<=0;last_duration_ms<=0;
        end else if(invalidate)begin
            have_previous<=0;armed<=0;active<=0;overlap_seen<=0;seen<=0;
            closed_mask<=0;known_mask<=0;left_current_ms<=0;right_current_ms<=0;
        end else if(update)begin
            have_previous<=1;previous_ms<=timestamp_ms;
            closed_mask<=next_closed;known_mask<=next_known;
            left_start<=start_l;right_start<=start_r;left_end<=end_l;right_end<=end_r;
            left_current_ms<=next_known[0] && next_closed[0] ? timestamp_ms-start_l : 0;
            right_current_ms<=next_known[1] && next_closed[1] ? timestamp_ms-start_r : 0;
            if(!continuous || next_known!=3)begin
                armed<=next_known==3 && next_closed==0;
                active<=0;overlap_seen<=0;seen<=0;
            end else if(next_closed==0)begin
                if(active && overlap_seen && duration_l>=MINIMUM_MS && duration_r>=MINIMUM_MS &&
                    overlap_end>=overlap_start && overlap_ms>=MINIMUM_MS)begin
                    count<=count+1'b1;event_pulse<=1;
                    if(duration_l>=LONG_MS || duration_r>=LONG_MS)long_count<=long_count+1'b1;
                    last_left_ms<=duration_l;last_right_ms<=duration_r;
                    last_overlap_ms<=overlap_ms;last_duration_ms<=timestamp_ms-union_start;
                end
                active<=0;overlap_seen<=0;seen<=0;armed<=1;
            end else if(armed && !active)begin
                active<=1;seen<=next_closed;union_start<=timestamp_ms;overlap_seen<=next_closed==3;
            end else if(active)begin
                if(|(seen & ~closed_mask & next_closed))begin active<=0;armed<=0;overlap_seen<=0;end
                else begin
                    seen<=seen | next_closed;
                    if(next_closed==3 && (onset_lr<=SYNC_MS || onset_rl<=SYNC_MS))overlap_seen<=1;
                end
            end
        end
    end
endmodule
