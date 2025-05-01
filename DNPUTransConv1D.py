from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU

import torch

class DNPUTransConv1D(DNPU):
    def __init__(
        self,
        processor : Processor,
        data_input_indices : list,
        in_channels : int,
        out_channels : int,
        kernel_size : int = 3,
        stride : int = 1,
        #padding: int = 0,
        dilation : int = 0,
        forward_pass_type : str = 'vec'
    ) -> None:
        
        super(DNPUTransConv1D, self).__init__(
            processor,
            data_input_indices,
            forward_pass_type = forward_pass_type
        )

        assert type(in_channels) is int, 'in_channels should be integer'
        assert type(out_channels) is int, 'out_channels should be integer'
        assert type(kernel_size) is int, 'kernel_size should be integer; e.g., 3'
        assert type(stride) is int, 'NOT SUPPORTED: in_channels should be integer'
        #assert type(padding) is int, 'NOT SUPPORTED: in_channels should be integer'
        assert type(dilation) is int, 'dilation should be integer'
        
        """
        The assert on data_input_indices had to change. The node number should now equal the kernel size and each node only has to contain a single input
        """
        #assert (
        #    torch.tensor(data_input_indices).numel() == kernel_size
        #), "Data input indices should be defined as mapping a single kernel. E.g., for a 1x3 1D-convolution you need 3 data input indices, represented as (dnpu_node_no=1, data_input_no_per_dnpu_node=3)."
        
        assert (
            torch.tensor(data_input_indices).size() == torch.Size([kernel_size, 1])
        ), "Data input indices should be defined as mapping a single kernel. E.g., for a 1x3 1D-tranposed convolution you need 3 single-input DNPU nodes, represented as (dnpu_node_no=3, data_input_no_per_dnpu_node=1)."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        #self.padding = padding
        self.stride = stride
        self.dilation = dilation
        
        """
        self.unfold has to change maybe but for now I removed it. Each column after the unfold is one input for the convolution
        but transposed concolution only has 1 input, so each column is already one input for the convolution
        """
        #self.unfold = torch.nn.Unfold(
        #    kernel_size = (1, self.kernel_size),
        #    stride = self.stride,
        #    padding = self.padding
        #)

        self.input_transform = True

        self.init_params()

    def init_params(self):
        """
        Initialises the control electrode indices and the data input electrode indices
        according to the size of the convolution. After that, reinitialises the control
        voltages (bias) that were initialised on the super call, but this time with the
        new dimensions for the control and data input indices.
        """
        # -- Setup node --
        control_shape = list(self.control_indices.shape)
        control_shape.insert(0, self.out_channels)
        control_shape.insert(0, self.in_channels)

        self.control_indices = self.control_indices.expand(
            control_shape).clone()

        control_shape.append(
            2)  # Extra dimension for minimum and maximum in control ranges
        self.control_ranges = self.control_ranges.expand(control_shape).clone()

        # -- Set everything as torch Tensors and send to DEVICE --
        data_input_shape = list(self.data_input_indices.shape)
        data_input_shape.insert(0, self.out_channels)
        data_input_shape.insert(0, self.in_channels)

        self.data_input_indices = self.data_input_indices.expand(
            data_input_shape).clone()
        # a method from DNPU parent; reinitializing the control voltages
        self.reset()


    """
    add_input_transpose: I don't yet know the purpose of this function nor have I seen an effect of this function
    I also don't see any foul in including it
    """
    def add_input_transform(self, input_range, strict=True):
        """
        Adds a linear transformation required to convert the input into the input electrode
        voltage ranges. It automatically calculates the input electrode voltage ranges for a
        particular DNPU according to the voltage ranges it was trained with. It is used typically
        to perform a current to voltage transformation, but it can also be applied for transforming
        the raw values from a dataset into voltages. The application of the input transformation
        occurs when the data has been reshaped into
        [Batch_size, window_no, in_chanels, node_no, input_electrode_no]. This function
        has to be called from outside the module, after its initialisation.

        Parameters
        ----------
        input_range : list
            The range that the original raw input data is going to have. It can be specified with
            two values [min, max], representing the minimum and maximum values that the input data
            is expected to have. In this case, the linear transformation will be adapted to the
            length of the input dimension automatically. E.g. input_range = [0,1].
            It can also be specified for different minimum and maximum value ranges per electrode.
            In this case, the list has to be specified with the same shape as the input_range
            variable of the class. This can be obtained by calling the get_input_ranges method.
        """
        super(DNPUTransConv1D, self).add_input_transform(input_range, strict=strict)

    def get_output_dim(self, dim):
        """
        Get the expected dimension of the output after the convolution.
        """
        return int(((dim + (2 * self.padding) - 1) * self.stride) + ((self.kernel_size * (self.dilation + 1)) - self.dilation))

    def preprocess(self, x):
        """
        It extracts sliding local blocks from a batched input tensor. Then, it reshapes the
        input in a vectorised way, so that the input has the following a shape of
        (batch_size, dnpu_electrode_no). It applies batch norm and/or a linear transformation
        if these are added by calling add_input_transform after the
        initialisation of this module. These call only needs to happen once.

        Parameters
        ----------
        x : torch.Tensor
            The raw input data to the convolution.

        Returns
        -------
        torch.Tensor

        """
        #x = self.unfold(x)
        x = x.squeeze(2)
        x = x.expand(x.shape[0], self.node_no, x.shape[2])


        # Transpose the window_size dimension by the window_no dimension
        x = x.transpose(1, 2)

        # Reshape as: [Batch_size, window_no, in_chanels, node_no, input_electrode_no],
        # where node_no is the number of DNPUs
        x = x.reshape(x.shape[0], x.shape[1], self.in_channels,
                      self.get_node_no(), self.get_data_input_electrode_no())

        if self.input_transform:
            self.add_input_transform([0., 1.], True)

        # Repeat info that will be used for each DNPU kernel
        # Shape as: [Batch_size, window_no, in_chanels, out_channels, node_no, input_electrode_no],
        # where node_no is the number of DNPUs.
        x = x.unsqueeze(3).expand(x.shape[0], x.shape[1], x.shape[2],
                                  self.out_channels, x.shape[3], x.shape[4])

        return x

    def merge_electrode_data(self, x):
        
        """
        Merge the input data to be fed to the input data electrodes with the
        data to be fed to the control voltage electrodes.

        Parameters
        ----------
        x: torch.tensor
            Input data that will be fed into the input data electrodes.

        Returns
        -------
        data: torch.Tensor
            A tensor with the input data and control voltage data to be fed
            through the activation electrodes, ordered according to the configurations
            of the indices for the data input and control voltage inputs to the
            DNPU convolution architecture. The data is given with a shape of:
            (batch_size,electrode_no).

        original_data_dim: torch.Size
            The original data dimensions of the data before being converted into a
            shape of (batch_size,electrode_no). This information is used to reconstruct
            the output tensor after is passed through the processor.  

        """
        # Expand controls according to batch_size and window_no
        controls_shape = list(self.control_voltages.shape)
        controls_shape.insert(0, x.shape[1])  # Add window_no dimension
        controls_shape.insert(0, x.shape[0])  # Add batch_size dimension

        controls = self.control_voltages.expand(controls_shape)

        # Expand indices according to batch size
        control_indices = self.control_indices.expand(controls_shape)
        input_indices = self.data_input_indices.expand_as(x)
        original_data_dim = x.shape

        # Create input data and order it according to the indices
        last_dim = len(controls.shape) - 1  # For concatenating purposes
        indices = torch.argsort(torch.cat((input_indices, control_indices),
                                          dim=last_dim),
                                dim=last_dim)

        data = torch.cat((x, controls), dim=last_dim)
        data = torch.gather(data, last_dim, indices)
        data = data.reshape(-1, data.shape[-1])

        return data, original_data_dim


    def summation(self, input: torch.tensor, output_dim: int):
        
        #Transposes the output of the DNPUs back to horizontal
        input = input.transpose(1, 3)

        #Interpolates with zeros, is done to achieve higher strides
        input = torch.nn.ZeroPad1d((0, self.stride - 1))(input.unsqueeze(4))
        input = input.reshape(input.shape[0],input.shape[1],input.shape[2],input.shape[3]*self.stride)

        #Zeropads the input so the skewing funtion works
        input = torch.nn.ZeroPad1d(self.kernel_size * (self.dilation + 1))(input)

        #Skews the input. The further it skews the higher the dilation
        input = input.as_strided((input.shape[0], input.shape[1], input.shape[2], output_dim + (self.dilation + 1)),
                                 (0, input.shape[3] + (self.dilation + 1), 0 ,1))

        #Sums the skewed input
        #After as_strided, (self.dilation + 1) amount of additional output columns are left at the front 
        #These columns are all zeros and should be removed
        return input.sum(1)[:, :, (self.dilation + 1):]


    def postprocess(self, result, data_dim, output_dim):
        
        result = result.reshape(data_dim[:-1])
        result = result.sum(dim=2)  # Sum values from the input kernels

        """
        New summation comes here
        """
        result = self.summation(result, output_dim)

        #result = result.transpose(
        #   1, 2)  # Return the output_kernel_no dimension to dimension 1.
        result = result.reshape(result.shape[0], result.shape[1], output_dim,
                                -1)
        return result

    def forward(self, x):
        """
        Forward pass of the convolution module.

        Parameters
        ----------
        x: torch.Tensor
            Input to the convolution, with a shape of
            (batch_size, channel_no, input_img_height, input_image_width)

        Returns
        -------
        result: torch.Tensor
            Image out of the convolution. With a shape of (batch_size, channel_no,
            output_feature_height, output_feature_width)
        """
        output_dim = self.get_output_dim(x.shape[3])
        x = self.preprocess(x)
        x, original_data_dim = self.merge_electrode_data(x)
        x = self.processor(x)   
        x = self.postprocess(x, original_data_dim, output_dim)

        return x.squeeze()