Ο
Ϋ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018χε
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0

output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:d*
dtype0
x
input_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_nameinput_layer/bias
q
$input_layer/bias/Read/ReadVariableOpReadVariableOpinput_layer/bias*
_output_shapes
:d*
dtype0

input_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*#
shared_nameinput_layer/kernel
y
&input_layer/kernel/Read/ReadVariableOpReadVariableOpinput_layer/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
π
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*«
value‘B B

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
 
0
1
2
3*
 
0
1
2
3*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
!trace_0
"trace_1
#trace_2
$trace_3* 
6
%trace_0
&trace_1
'trace_2
(trace_3* 
* 
:
)iter
	*decay
+learning_rate
,momentum*

-serving_default* 

0
1*

0
1*
* 

.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

3trace_0* 

4trace_0* 
b\
VARIABLE_VALUEinput_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEinput_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

:trace_0* 

;trace_0* 
c]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

<0
=1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
KE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
>	variables
?	keras_api
	@total
	Acount*
H
B	variables
C	keras_api
	Dtotal
	Ecount
F
_fn_kwargs*

@0
A1*

>	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

D0
E1*

B	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

!serving_default_input_layer_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

StatefulPartitionedCallStatefulPartitionedCall!serving_default_input_layer_inputinput_layer/kernelinput_layer/biasoutput_layer/kerneloutput_layer/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_10305
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ί
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&input_layer/kernel/Read/ReadVariableOp$input_layer/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_10460
Ε
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput_layer/kernelinput_layer/biasoutput_layer/kerneloutput_layer/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_10506ά¨
½
Ϊ
,__inference_sequential_1_layer_call_fn_10185
input_layer_input
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_nameinput_layer_input
Ι	
χ
F__inference_input_layer_layer_call_and_return_conditional_losses_10151

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
ξ
G__inference_sequential_1_layer_call_and_return_conditional_losses_10363

inputs<
*input_layer_matmul_readvariableop_resource:d9
+input_layer_biasadd_readvariableop_resource:d=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identity’"input_layer/BiasAdd/ReadVariableOp’!input_layer/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
output_layer/MatMulMatMulinput_layer/BiasAdd:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
IdentityIdentityoutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ϊ
NoOpNoOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ	
ψ
G__inference_output_layer_layer_call_and_return_conditional_losses_10401

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ζ

+__inference_input_layer_layer_call_fn_10372

inputs
unknown:d
	unknown_0:d
identity’StatefulPartitionedCallΫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_input_layer_layer_call_and_return_conditional_losses_10151o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι	
χ
F__inference_input_layer_layer_call_and_return_conditional_losses_10382

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:d
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Κ	
ψ
G__inference_output_layer_layer_call_and_return_conditional_losses_10167

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs

Ο
,__inference_sequential_1_layer_call_fn_10331

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Η
ξ
G__inference_sequential_1_layer_call_and_return_conditional_losses_10347

inputs<
*input_layer_matmul_readvariableop_resource:d9
+input_layer_biasadd_readvariableop_resource:d=
+output_layer_matmul_readvariableop_resource:d:
,output_layer_biasadd_readvariableop_resource:
identity’"input_layer/BiasAdd/ReadVariableOp’!input_layer/MatMul/ReadVariableOp’#output_layer/BiasAdd/ReadVariableOp’"output_layer/MatMul/ReadVariableOp
!input_layer/MatMul/ReadVariableOpReadVariableOp*input_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
input_layer/MatMulMatMulinputs)input_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
"input_layer/BiasAdd/ReadVariableOpReadVariableOp+input_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0
input_layer/BiasAddBiasAddinput_layer/MatMul:product:0*input_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
output_layer/MatMulMatMulinput_layer/BiasAdd:output:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
IdentityIdentityoutput_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ϊ
NoOpNoOp#^input_layer/BiasAdd/ReadVariableOp"^input_layer/MatMul/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2H
"input_layer/BiasAdd/ReadVariableOp"input_layer/BiasAdd/ReadVariableOp2F
!input_layer/MatMul/ReadVariableOp!input_layer/MatMul/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Θ1
ά
!__inference__traced_restore_10506
file_prefix5
#assignvariableop_input_layer_kernel:d1
#assignvariableop_1_input_layer_bias:d8
&assignvariableop_2_output_layer_kernel:d2
$assignvariableop_3_output_layer_bias:%
assignvariableop_4_sgd_iter:	 &
assignvariableop_5_sgd_decay: .
$assignvariableop_6_sgd_learning_rate: )
assignvariableop_7_sgd_momentum: $
assignvariableop_8_total_1: $
assignvariableop_9_count_1: #
assignvariableop_10_total: #
assignvariableop_11_count: 
identity_13’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ϋ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*‘
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B ί
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_input_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_input_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp&assignvariableop_2_output_layer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp$assignvariableop_3_output_layer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Χ
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: Δ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
½
Ϊ
,__inference_sequential_1_layer_call_fn_10258
input_layer_input
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10234o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_nameinput_layer_input
?
Μ
G__inference_sequential_1_layer_call_and_return_conditional_losses_10272
input_layer_input#
input_layer_10261:d
input_layer_10263:d$
output_layer_10266:d 
output_layer_10268:
identity’#input_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputinput_layer_10261input_layer_10263*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_input_layer_layer_call_and_return_conditional_losses_10151£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0output_layer_10266output_layer_10268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_10167|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_nameinput_layer_input
ή
Α
G__inference_sequential_1_layer_call_and_return_conditional_losses_10234

inputs#
input_layer_10223:d
input_layer_10225:d$
output_layer_10228:d 
output_layer_10230:
identity’#input_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallω
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_10223input_layer_10225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_input_layer_layer_call_and_return_conditional_losses_10151£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0output_layer_10228output_layer_10230*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_10167|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι
Ί
 __inference__wrapped_model_10134
input_layer_inputI
7sequential_1_input_layer_matmul_readvariableop_resource:dF
8sequential_1_input_layer_biasadd_readvariableop_resource:dJ
8sequential_1_output_layer_matmul_readvariableop_resource:dG
9sequential_1_output_layer_biasadd_readvariableop_resource:
identity’/sequential_1/input_layer/BiasAdd/ReadVariableOp’.sequential_1/input_layer/MatMul/ReadVariableOp’0sequential_1/output_layer/BiasAdd/ReadVariableOp’/sequential_1/output_layer/MatMul/ReadVariableOp¦
.sequential_1/input_layer/MatMul/ReadVariableOpReadVariableOp7sequential_1_input_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0¦
sequential_1/input_layer/MatMulMatMulinput_layer_input6sequential_1/input_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d€
/sequential_1/input_layer/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_input_layer_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0Α
 sequential_1/input_layer/BiasAddBiasAdd)sequential_1/input_layer/MatMul:product:07sequential_1/input_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d¨
/sequential_1/output_layer/MatMul/ReadVariableOpReadVariableOp8sequential_1_output_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0ΐ
 sequential_1/output_layer/MatMulMatMul)sequential_1/input_layer/BiasAdd:output:07sequential_1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????¦
0sequential_1/output_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Δ
!sequential_1/output_layer/BiasAddBiasAdd*sequential_1/output_layer/MatMul:product:08sequential_1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????y
IdentityIdentity*sequential_1/output_layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp0^sequential_1/input_layer/BiasAdd/ReadVariableOp/^sequential_1/input_layer/MatMul/ReadVariableOp1^sequential_1/output_layer/BiasAdd/ReadVariableOp0^sequential_1/output_layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2b
/sequential_1/input_layer/BiasAdd/ReadVariableOp/sequential_1/input_layer/BiasAdd/ReadVariableOp2`
.sequential_1/input_layer/MatMul/ReadVariableOp.sequential_1/input_layer/MatMul/ReadVariableOp2d
0sequential_1/output_layer/BiasAdd/ReadVariableOp0sequential_1/output_layer/BiasAdd/ReadVariableOp2b
/sequential_1/output_layer/MatMul/ReadVariableOp/sequential_1/output_layer/MatMul/ReadVariableOp:Z V
'
_output_shapes
:?????????
+
_user_specified_nameinput_layer_input

Ρ
#__inference_signature_wrapper_10305
input_layer_input
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_10134o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_nameinput_layer_input
Τ!

__inference__traced_save_10460
file_prefix1
-savev2_input_layer_kernel_read_readvariableop/
+savev2_input_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ψ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*‘
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B €
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_input_layer_kernel_read_readvariableop+savev2_input_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*G
_input_shapes6
4: :d:d:d:: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ή
Α
G__inference_sequential_1_layer_call_and_return_conditional_losses_10174

inputs#
input_layer_10152:d
input_layer_10154:d$
output_layer_10168:d 
output_layer_10170:
identity’#input_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCallω
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinputsinput_layer_10152input_layer_10154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_input_layer_layer_call_and_return_conditional_losses_10151£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0output_layer_10168output_layer_10170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_10167|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Μ
G__inference_sequential_1_layer_call_and_return_conditional_losses_10286
input_layer_input#
input_layer_10275:d
input_layer_10277:d$
output_layer_10280:d 
output_layer_10282:
identity’#input_layer/StatefulPartitionedCall’$output_layer/StatefulPartitionedCall
#input_layer/StatefulPartitionedCallStatefulPartitionedCallinput_layer_inputinput_layer_10275input_layer_10277*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_input_layer_layer_call_and_return_conditional_losses_10151£
$output_layer/StatefulPartitionedCallStatefulPartitionedCall,input_layer/StatefulPartitionedCall:output:0output_layer_10280output_layer_10282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_10167|
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
NoOpNoOp$^input_layer/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2J
#input_layer/StatefulPartitionedCall#input_layer/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_nameinput_layer_input

Ο
,__inference_sequential_1_layer_call_fn_10318

inputs
unknown:d
	unknown_0:d
	unknown_1:d
	unknown_2:
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_10174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Θ

,__inference_output_layer_layer_call_fn_10391

inputs
unknown:d
	unknown_0:
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_10167o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs"ΏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Γ
serving_default―
O
input_layer_input:
#serving_default_input_layer_input:0?????????@
output_layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Y
΄
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ζ
!trace_0
"trace_1
#trace_2
$trace_32ϋ
,__inference_sequential_1_layer_call_fn_10185
,__inference_sequential_1_layer_call_fn_10318
,__inference_sequential_1_layer_call_fn_10331
,__inference_sequential_1_layer_call_fn_10258ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z!trace_0z"trace_1z#trace_2z$trace_3
?
%trace_0
&trace_1
'trace_2
(trace_32η
G__inference_sequential_1_layer_call_and_return_conditional_losses_10347
G__inference_sequential_1_layer_call_and_return_conditional_losses_10363
G__inference_sequential_1_layer_call_and_return_conditional_losses_10272
G__inference_sequential_1_layer_call_and_return_conditional_losses_10286ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z%trace_0z&trace_1z'trace_2z(trace_3
ΥB?
 __inference__wrapped_model_10134input_layer_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
I
)iter
	*decay
+learning_rate
,momentum"
	optimizer
,
-serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ο
3trace_02?
+__inference_input_layer_layer_call_fn_10372’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z3trace_0

4trace_02ν
F__inference_input_layer_layer_call_and_return_conditional_losses_10382’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z4trace_0
$:"d2input_layer/kernel
:d2input_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
π
:trace_02Σ
,__inference_output_layer_layer_call_fn_10391’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z:trace_0

;trace_02ξ
G__inference_output_layer_layer_call_and_return_conditional_losses_10401’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z;trace_0
%:#d2output_layer/kernel
:2output_layer/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_1_layer_call_fn_10185input_layer_input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ώBϋ
,__inference_sequential_1_layer_call_fn_10318inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ώBϋ
,__inference_sequential_1_layer_call_fn_10331inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
,__inference_sequential_1_layer_call_fn_10258input_layer_input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_10347inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_10363inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
€B‘
G__inference_sequential_1_layer_call_and_return_conditional_losses_10272input_layer_input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
€B‘
G__inference_sequential_1_layer_call_and_return_conditional_losses_10286input_layer_input"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
ΤBΡ
#__inference_signature_wrapper_10305input_layer_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ίBά
+__inference_input_layer_layer_call_fn_10372inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϊBχ
F__inference_input_layer_layer_call_and_return_conditional_losses_10382inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΰBέ
,__inference_output_layer_layer_call_fn_10391inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ϋBψ
G__inference_output_layer_layer_call_and_return_conditional_losses_10401inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
N
>	variables
?	keras_api
	@total
	Acount"
_tf_keras_metric
^
B	variables
C	keras_api
	Dtotal
	Ecount
F
_fn_kwargs"
_tf_keras_metric
.
@0
A1"
trackable_list_wrapper
-
>	variables"
_generic_user_object
:  (2total
:  (2count
.
D0
E1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper£
 __inference__wrapped_model_10134:’7
0’-
+(
input_layer_input?????????
ͺ ";ͺ8
6
output_layer&#
output_layer?????????¦
F__inference_input_layer_layer_call_and_return_conditional_losses_10382\/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????d
 ~
+__inference_input_layer_layer_call_fn_10372O/’,
%’"
 
inputs?????????
ͺ "?????????d§
G__inference_output_layer_layer_call_and_return_conditional_losses_10401\/’,
%’"
 
inputs?????????d
ͺ "%’"

0?????????
 
,__inference_output_layer_layer_call_fn_10391O/’,
%’"
 
inputs?????????d
ͺ "?????????Ό
G__inference_sequential_1_layer_call_and_return_conditional_losses_10272qB’?
8’5
+(
input_layer_input?????????
p 

 
ͺ "%’"

0?????????
 Ό
G__inference_sequential_1_layer_call_and_return_conditional_losses_10286qB’?
8’5
+(
input_layer_input?????????
p

 
ͺ "%’"

0?????????
 ±
G__inference_sequential_1_layer_call_and_return_conditional_losses_10347f7’4
-’*
 
inputs?????????
p 

 
ͺ "%’"

0?????????
 ±
G__inference_sequential_1_layer_call_and_return_conditional_losses_10363f7’4
-’*
 
inputs?????????
p

 
ͺ "%’"

0?????????
 
,__inference_sequential_1_layer_call_fn_10185dB’?
8’5
+(
input_layer_input?????????
p 

 
ͺ "?????????
,__inference_sequential_1_layer_call_fn_10258dB’?
8’5
+(
input_layer_input?????????
p

 
ͺ "?????????
,__inference_sequential_1_layer_call_fn_10318Y7’4
-’*
 
inputs?????????
p 

 
ͺ "?????????
,__inference_sequential_1_layer_call_fn_10331Y7’4
-’*
 
inputs?????????
p

 
ͺ "?????????Ό
#__inference_signature_wrapper_10305O’L
’ 
EͺB
@
input_layer_input+(
input_layer_input?????????";ͺ8
6
output_layer&#
output_layer?????????