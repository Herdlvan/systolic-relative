module sync_fifo #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16
)(
    input clk,
    input rst_n,
    input wr_en,
    input rd_en,
    input [DATA_WIDTH-1:0] din,
    output reg [DATA_WIDTH-1:0] dout,
    output empty,
    output full
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] wptr, rptr, count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= 0; rptr <= 0; count <= 0;
            empty <= 1; full <= 0;
        end else begin
            // 写
            if (wr_en && !full) begin
                mem[wptr] <= din;
                wptr <= (wptr + 1) % DEPTH;
                count <= count + 1;
            end
            // 读
            if (rd_en && !empty) begin
                dout <= mem[rptr];
                rptr <= (rptr + 1) % DEPTH;
                count <= count - 1;
            end
        end
    end


         assign empty= (count == 0);
         assign full = (count == DEPTH);

endmodule