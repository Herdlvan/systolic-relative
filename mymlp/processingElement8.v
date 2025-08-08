`timescale 100 ns / 10 ps

module processingElement8(clk,reset,floatA,floatB,result);

parameter DATA_WIDTH = 8;

input clk, reset;
input [DATA_WIDTH-1:0] floatA, floatB;
output reg [DATA_WIDTH-1:0] result;

wire [DATA_WIDTH-1:0] multResult;
wire [DATA_WIDTH-1:0] addResult;

int8Mul FM (floatA,floatB,multResult);
int8Add FADD (multResult,result,addResult);

always @ (posedge clk or posedge reset) begin
	if (reset == 1'b1) begin
		result = 0;
	end else begin
		result = addResult;
	end
end

endmodule
