module embedded_table#(
 parameter DATA_WIDTH=8
)(
input wire clk,
input read_enable,
input wire [9:0] index,
output reg [DATA_WIDTH*8-1:0] data_out 
);

reg [DATA_WIDTH-1:0]embedding[1024:0][7:0];

initial begin
        $readmemh("/Users/boysdontcry104/Documents/vscode/科研/module/quantized_weights_signed/quantized_embedding_M.txt", embedding);
    end

always@(posedge clk)begin
	if(!read_enable) begin
		data_out<=0;
	end
	else begin
		data_out[DATA_WIDTH-1:0]<=embedding[index][0];
		data_out[2*DATA_WIDTH-1:DATA_WIDTH]<=embedding[index][1];
		data_out[3*DATA_WIDTH-1:2*DATA_WIDTH]<=embedding[index][2];
		data_out[4*DATA_WIDTH-1:3*DATA_WIDTH]<=embedding[index][3];
		data_out[5*DATA_WIDTH-1:4*DATA_WIDTH]<=embedding[index][4];
		data_out[6*DATA_WIDTH-1:5*DATA_WIDTH]<=embedding[index][5];
		data_out[7*DATA_WIDTH-1:6*DATA_WIDTH]<=embedding[index][6];
		data_out[8*DATA_WIDTH-1:7*DATA_WIDTH]<=embedding[index][7];
		
	end

end
endmodule