Flow(
  (_transform): CompositeTransform(
    (_transforms): ModuleList(
      (0): AffineScalarTransform()
      (1): MultiscaleCompositeTransform(
        (_transforms): ModuleList(
          (0): CompositeTransform(
            (_transforms): ModuleList(
              (0): SqueezeTransform()
              (1): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (2): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (3): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (4): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (5): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (6): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (7): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(6, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 90, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (8): OneByOneConvolution(
                (permutation): RandomPermutation()
              )
            )
          )
          (1): CompositeTransform(
            (_transforms): ModuleList(
              (0): SqueezeTransform()
              (1): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (2): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (3): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (4): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (5): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (6): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (7): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(12, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 180, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (8): OneByOneConvolution(
                (permutation): RandomPermutation()
              )
            )
          )
          (2): CompositeTransform(
            (_transforms): ModuleList(
              (0): SqueezeTransform()
              (1): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (2): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (3): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (4): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (5): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (6): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (7): CompositeTransform(
                (_transforms): ModuleList(
                  (0): ActNorm()
                  (1): OneByOneConvolution(
                    (permutation): RandomPermutation()
                  )
                  (2): PiecewiseRationalLinearCouplingTransform(
                    (transform_net): ConvResidualNet(
                      (initial_layer): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                      (blocks): ModuleList(
                        (0): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (1): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                        (2): ConvResidualBlock(
                          (batch_norm_layers): ModuleList(
                            (0): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                            (1): BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
                          )
                          (conv_layers): ModuleList(
                            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                            (1): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                          )
                          (dropout): Dropout(p=0.2, inplace=False)
                        )
                      )
                      (final_layer): Conv2d(96, 360, kernel_size=(1, 1), stride=(1, 1))
                    )
                  )
                )
              )
              (8): OneByOneConvolution(
                (permutation): RandomPermutation()
              )
            )
          )
        )
      )
    )
  )
  (_distribution): StandardNormal()
)