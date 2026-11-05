// Completed per-eye OPEN->CLOSED->OPEN events, using host capture milliseconds.
// Timestamp differences are modulo 2^32; discontinuities cancel pending events.
module blink_event #(parameter integer MINIMUM_MS=30,MAXIMUM_MS=600,MAXIMUM_GAP_MS=250)(
    input wire clk, input wire reset, input wire invalidate,
    input wire update, input wire valid, input wire closed,
    input wire [31:0] timestamp_ms,
    output reg event_pulse=0, output reg [31:0] count=0);
    reg have_previous=0,previous_closed=0,armed=0;
    reg [31:0] previous_ms=0,closed_at=0;
    wire [31:0] gap=timestamp_ms-previous_ms;
    wire [31:0] duration=timestamp_ms-closed_at;
    always @(posedge clk) begin
        event_pulse<=0;
        if(reset)begin
            count<=0;have_previous<=0;previous_closed<=0;armed<=0;
            previous_ms<=0;closed_at<=0;
        end else if(invalidate)begin
            have_previous<=0;armed<=0;
        end else if(update)begin
            previous_ms<=timestamp_ms;previous_closed<=closed;
            have_previous<=valid;
            if(!valid || !have_previous || gap==0 || gap>MAXIMUM_GAP_MS)armed<=0;
            else if(!previous_closed && closed)begin armed<=1;closed_at<=timestamp_ms;end
            else if(previous_closed && !closed)begin
                armed<=0;
                if(armed && duration>=MINIMUM_MS && duration<=MAXIMUM_MS)begin
                    count<=count+1'b1;event_pulse<=1;
                end
            end
        end
    end
endmodule
