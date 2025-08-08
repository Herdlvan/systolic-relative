    module concat #(
    parameter DATA_WIDTH = 8
) (
    input clk,
    input [DATA_WIDTH*8-1:0] data_out_M,
    input [DATA_WIDTH*8-1:0] data_out_N,
    input [DATA_WIDTH*8-1:0] data_out_K,
    output reg [24 * DATA_WIDTH - 1:0] concat_out 
);

always @(*) begin
    concat_out = {data_out_M, data_out_N, data_out_K};
end

endmodule