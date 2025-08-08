`timescale 100 ns / 10 ps

module int8Add (
    input  signed [7:0] a,
    input  signed [7:0] b,
    output reg signed [7:0] sum
);

wire signed [8:0] add_result;
assign add_result = a + b;

always @(*) begin
    if (add_result > 127)
        sum = 127;
    else if (add_result < -128)
        sum = -128;
    else
        sum = add_result[7:0];
end

endmodule