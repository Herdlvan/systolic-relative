module layer1(clk,reset,input_fc,weights,output_fc);

parameter DATA_WIDTH = 8;
parameter INPUT_NODES = 24;      // 修改为24
parameter OUTPUT_NODES = 128;    // 修改为128

input clk, reset;
input [DATA_WIDTH*INPUT_NODES-1:0] input_fc;
input [DATA_WIDTH*OUTPUT_NODES-1:0] weights;
output [DATA_WIDTH*OUTPUT_NODES-1:0] output_fc;
output [4:0]weight_address1; 
reg [DATA_WIDTH-1:0] selectedInput;
integer j;

genvar i;

generate
    for (i = 0; i < OUTPUT_NODES; i = i + 1) begin
        processingElement8 PE 
        (
            .clk(clk),
            .reset(reset),
            .floatA(selectedInput),
            .floatB(weights[DATA_WIDTH*i+:DATA_WIDTH]),
            .result(output_fc[DATA_WIDTH*i+:DATA_WIDTH])
        );
    end
endgenerate
//input logic 
always @ (posedge clk or posedge reset) begin
    if (reset == 1'b1) begin
        selectedInput = 0;
        j = INPUT_NODES - 1;
    end else if (j < 0) begin
        selectedInput = 0;
    end else begin
        selectedInput = input_fc[DATA_WIDTH*j+:DATA_WIDTH];
        j = j - 1;
    end
end


always @(posedge clk or posedge reset) begin
    if (reset == 1'b1) begin
        weight_address1 =0;
end else if(weight_load)begin       
        if(weight_address1 < INPUT_NODES - 1) begin
            weight_address1 = weight_address1 + 1;
        end else begin
            weight_address1 = 0; 
        end
    end

    
end


endmodule