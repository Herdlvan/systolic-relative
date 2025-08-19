module PE_Unit (
    input clk,
    input reset,
    // 数据输入
    input [DATA_WIDTH-1:0] data_in_bottom,  
    input [DATA_WIDTH-1:0] data_in_left,  
    input [DATA_WIDTH-1:0] data_in_right,   
    input [DATA_WIDTH-1:0] data_in_top,
    input [DATA_WIDTH-1:0] Stationary_data,
    // 数据输出
    output [DATA_WIDTH-1:0] data_out_bottom,
    output [DATA_WIDTH-1:0] data_out_left,
    output [DATA_WIDTH-1:0] data_out_right,
    output [DATA_WIDTH-1:0] data_out_top,
    // 控制信号
    input [1:0] ctrl_crossbar [0:7],  // 8个Crossbar控制
    input [1:0] mux_sel,
    input mac_enable,                  // MAC使能信号
    input accum_clear,                  // 累加器清零
    input output_stationary_enable // 稳态reg使能信号
   
);

parameter DATA_WIDTH = 8;       // 数据位宽
parameter ACCUM_WIDTH = 16;      // 累加器位宽（防溢出）

//------------------------------------------
// 1. Crossbar网络（8个2x2 Crossbar）
//------------------------------------------
wire [DATA_WIDTH-1:0] cb0_out0, cb0_out1;
wire [DATA_WIDTH-1:0] cb1_out0, cb1_out1;
wire [DATA_WIDTH-1:0] cb2_in0, cb2_in1;
wire [DATA_WIDTH-1:0] cb3_in0, cb3_in1;
wire [DATA_WIDTH-1:0] cb4_out0, cb4_out1;
wire [DATA_WIDTH-1:0] cb5_out0, cb5_out1;
wire [DATA_WIDTH-1:0] cb6_out0, cb6_out1;
wire [DATA_WIDTH-1:0] cb7_out0, cb7_out1;
wire [ACCUM_WIDTH-1:0] accum;  // 累加器

reg [ACCUM_WIDTH-1:0]Stationary_data_reg;
reg [DATA_WIDTH-1:0]cb2_in0_r;
reg [DATA_WIDTH-1:0]cb2_in1_r;
reg [DATA_WIDTH-1:0]cb3_in0_r;
reg [DATA_WIDTH-1:0]cb3_in1_r;

assign cb2_in0 = cb2_in0_r;
assign cb2_in1 = cb2_in1_r;
assign cb3_in0 = cb3_in0_r; 
assign cb3_in1 = cb3_in1_r;  


// Crossbar实例化
// Crossbar 0: 连接 right和 left 输入
Crossbar2x2 crossbar_0 (
    .in0(data_in_left),
    .in1(data_in_right),
    .out0(cb0_out0), //mac
    .out1(cb0_out1),//pass-through
    .sel(ctrl_crossbar[0])
);
// Crossbar 1: 连接top和 bottom 输入
Crossbar2x2 crossbar_1 (
    .in0(data_in_top),
    .in1(data_in_bottom),
    .out0(cb1_out0), //mac
    .out1(cb1_out1),//pass-through
    .sel(ctrl_crossbar[1])
);

// Crossbar 2: 连接 right和 left output
Crossbar2x2 crossbar_2 (
    .in0(cb2_in0),
    .in1(cb2_in1),
    .out0(data_out_left), 
    .out1(data_out_right),
    .sel(ctrl_crossbar[2])
);
// Crossbar3: 连接  top和bottom  output
Crossbar2x2 crossbar_3 (
    .in0(cb3_in0),
    .in1(cb3_in1),
    .out0(data_out_top), 
    .out1(data_out_bottom),
    .sel(ctrl_crossbar[3])
);

// Crossbar 4: 连接MAC
Crossbar2x2 crossbar_4 (
    .in0(cb0_out0),
    .in1(cb1_out0),
    .out0(cb4_out0), 
    .out1(cb4_out1),
    .sel(ctrl_crossbar[4])
);
// Crossbar 5: 连接MAC and  statinary data
Crossbar2x2 crossbar_5 (
    .in0(cb4_out1),
    .in1(Stationary_data_reg[7:0]), // 只取低8位
    .out0(cb5_out0), 
    .out1(cb5_out1),
    .sel(ctrl_crossbar[5])
);

// Crossbar6:   reorder mac operand
Crossbar2x2 crossbar_6 (
    .in0(cb4_out0),
    .in1(accum[7:0]),
    .out0(cb6_out0), 
    .out1(cb6_out1),
    .sel(ctrl_crossbar[6])
);


// Crossbar7:   pass-through  turn or straight
Crossbar2x2 crossbar_7 (
    .in0(cb0_out1),
    .in1(cb1_out1),
    .out0(cb7_out0), 
    .out1(cb7_out1),
    .sel(ctrl_crossbar[7])
);



//------------------------------------------
// 2. MAC单元输入重排序
//------------------------------------------
wire [DATA_WIDTH-1:0] mac_a, mac_b, mac_c;
assign mac_a =cb4_out0 ;  // 乘法操作数1
assign mac_b = cb5_out0;  // 乘法操作数2
assign mac_c = cb5_out1;  // 累加操作数

//------------------------------------------
// 3. MAC计算
//------------------------------------------

// 乘法器
wire [ACCUM_WIDTH-1:0] mult_result;
assign mult_result = mac_a * mac_b;  // 有符号乘法
assign accum=mult_result + mac_c;


// Statinary_data_reg update（时序逻辑）
//if output stationary,then update reg Statinary_data_reg update   others,keep the original value./
always @(posedge clk or posedge reset) begin
        if (reset) begin
                Stationary_data_reg <= 0; // 清零
        end
        else if(mac_enable) begin
             if (output_stationary_enable) begin
                Stationary_data_reg <= accum; // 更新为当前累加值              
            end
            else begin
            Stationary_data_reg[7:0]<= Stationary_data; // 保持原值
            end
        end
end
//------------------------------------------
// 4. pipline output
//------------------------------------------
always @(posedge clk or posedge reset) begin
    if (reset) begin
        cb2_in0_r<=0;
        cb2_in1_r<=0;
        cb3_in0_r<=0;
        cb3_in1_r<=0;
    end
   else  begin
        cb2_in0_r<=MUX0_out;
        cb2_in1_r<=cb7_out0;
        cb3_in0_r<=MUX1_out;
        cb3_in1_r<=cb7_out1;
        
    end
end

assign MUX0_out=(mux_sel[0]==1)?cb0_out0:cb6_out1;
assign MUX1_out=(mux_sel[1]==1)?cb1_out0:cb6_out0;




endmodule