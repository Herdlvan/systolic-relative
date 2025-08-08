`timescale 100 ns / 10 ps

module int8Mul (
    input  signed [7:0] floatA,
    input  signed [7:0] floatB,
    output reg signed [7:0] product
);

wire signed [15:0] mult_result;
assign mult_result = floatA * floatB;

always @(*) begin
    if (mult_result > 127)
        product = 127;
    else if (mult_result < -128)
        product = -128;
    else
        product = mult_result[7:0];
end

endmodule