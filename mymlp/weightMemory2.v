module weightMemory2(clk,address,weights);

parameter DATA_WIDTH = 8;
parameter INPUT_NODES = 128;
parameter OUTPUT_NODES = 387;
parameter file = "/Users/boysdontcry104/Documents/vscode/科研/module/quantized_weights_signed/quantized_fc2.weight.txt";

localparam TOTAL_WEIGHT_SIZE = INPUT_NODES * OUTPUT_NODES;

input clk;
input [7:0] address;
output reg [DATA_WIDTH*OUTPUT_NODES-1:0] weights;

reg [DATA_WIDTH-1:0] memory [0:TOTAL_WEIGHT_SIZE-1];

integer i;

always @ (posedge clk) begin	
	if (address > INPUT_NODES-1 || address < 0) begin
		weights = 0;
	end else begin
		for (i = 0; i < OUTPUT_NODES; i = i + 1) begin
			weights[(OUTPUT_NODES-1-i)*DATA_WIDTH+:DATA_WIDTH] = memory[(address*OUTPUT_NODES)+i];
		end
	end
end

initial begin
	$readmemh(file,memory);
end

endmodule
