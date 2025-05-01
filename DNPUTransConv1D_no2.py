from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU

import torch

class DNPUTransConv1D_no2(DNPU):
    def __init__(
            self,
            processor : Processor,
            data_input_indices : list,
            in_channels : int,
            out_channels : int,
            kernel_size : int = 3,
            stride : int = 1,
            padding: int = 0,
            interpolating: int = 0,
            forward_pass_type : str = 'vec'
        ) -> None:
        super(DNPUTransConv1D_no2, self).__init__(
            processor,
            data_input_indices,
            forward_pass_type = forward_pass_type
        )
        assert type(in_channels) is int, 'in_channels should be integer'
        assert type(out_channels) is int, 'out_channels should be integer'
        assert type(
            kernel_size 
        ) is int, 'kernel_size should be integer; e.g., 3'
        assert type(stride) is int, 'NOT SUPPORTED: in_channels should be integer'
        assert type(padding) is int, 'NOT SUPPORTED: in_channels should be integer'
        assert type(interpolating) is int, 'interpolating shoeld be integer'
        assert (
            torch.tensor(data_input_indices).numel() == kernel_size
        ), "Data input indices should be defined as mapping a single kernel. E.g., for a 1x3 1D-convolution you need 3 data input indices, represented as (dnpu_node_no=1, data_input_no_per_dnpu_node=3)."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.interpolating = interpolating
        # supports only 4-D (image-like) tensors, so for audio, I used (1, self.kernel_size)
        self.unfold = torch.nn.Unfold(
            kernel_size = (1, self.kernel_size)
        )

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
        super(DNPUTransConv1D_no2, self).add_input_transform(input_range, strict=strict)

    def get_output_dim(self, dim):
        """
        Get the expected dimension of the output after the convolution.
        """ 
        return int((((dim * (self.interpolating + 1)) +
                     (2 * self.padding) - self.kernel_size) / self.stride) + 1)
    
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
        # Interpolates x with zeros
        x = torch.nn.ZeroPad1d((0, self.interpolating))(x.unsqueeze(4))
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3] * (self.interpolating + 1))

        # Zero pads x
        x = torch.nn.ZeroPad1d(self.padding)(x)

        x = self.unfold(x)

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

    def postprocess(self, result, data_dim, output_dim):
        """
        The shape of the output of the convolution after passing through the processor is of
        (batch_size,electrode_no). This method does the final operations of the convolution,
        and to make the output have the same data shape as it would from outside a covolution.

        The postprocessing sums the values from the input kernel dimensions, and then applies
        either a sum or a linear operation to combine the outputs of the DNPU Convolution module.

        Parameters
        ----------
        result: torch.Tensor
            A tensor with the input data and control voltage data to be fed
            through the activation electrodes, ordered according to the configurations
            of the indices for the data input and control voltage inputs to the
            DNPU convolution architecture. The data is given with a shape of:
            (batch_size,electrode_no).

        data_dim: torch.Shape
            The original data dimensions of the data before being converted into a
            shape of (batch_size,electrode_no). This information is used to reconstruct
            the output tensor after is passed through the processor.

        output_dim: int
            Dimension of the output after the convolution. It can be calculated with the method
            get_output_dim of this module.

        Returns
        -------
        result: torch.Tensor
            Image out of the convolution. With a shape of (batch_size, channel_no,
            output_feature_height, output_feature_width)
        """
        result = result.reshape(data_dim[:-1])
        result = result.sum(dim=2)  # Sum values from the input kernels

        result = result.sum(
            dim=3)  # Sum the output from the devices used for the convolution
        # result = result.squeeze()

        result = result.transpose(
            1, 2)  # Return the output_kernel_no dimension to dimension 1.
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
