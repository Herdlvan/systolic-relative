
module Crossbar2x2#(
parameter DATA_WIDTH = 8) (
    input [DATA_WIDTH-1:0] in0,
    input [DATA_WIDTH-1:0] in1,
    output reg [DATA_WIDTH-1:0] out0,
    output reg [DATA_WIDTH-1:0] out1,
    input [1:0] sel
);

always @(*) begin
    case (sel)
        2'b00: begin out0 = in0; out1 = in1; end // 直通
        2'b01: begin out0 = in1; out1 = in0; end // 交叉
        2'b10: begin out0 = in0; out1 = in0; end // 广播 in0
        2'b11: begin out0 = in1; out1 = in1; end // 广播 in1
    endcase
end
endmodule