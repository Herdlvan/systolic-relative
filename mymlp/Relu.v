module Relu #(
    parameter DATA_WIDTH =8
) (
    input clk,
    input rst_n,
    input [DATA_WIDTH-1:0] in_data,
    output reg [DATA_WIDTH-1:0]out_data

);

always @(posedge clk) begin
    if(rst_n==1)
        out_data<=0;
    else begin
        if(in_data>0)
            out_data<=in_data;
        else
            out_data<=0;
    end
end


endmodule