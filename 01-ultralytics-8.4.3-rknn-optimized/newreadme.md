
## 修改ultralytics/nn/autobackend.py文件
576-591行             

    # atk-add
            assert "for inference, please refer to https://github.com/airockchip/rknn_model_zoo/"
            # if not is_rockchip():
            #     raise OSError("RKNN inference is only supported on Rockchip devices.")
            # LOGGER.info(f"Loading {w} for RKNN inference...")
            # check_requirements("rknn-toolkit-lite2")
            # from rknnlite.api import RKNNLite

            # w = Path(w)
            # if not w.is_file():  # if not *.rknn
            #     w = next(w.rglob("*.rknn"))  # get *.rknn file from *_rknn_model dir
            # rknn_model = RKNNLite()
            # rknn_model.load_rknn(str(w))
            # rknn_model.init_runtime()
            # metadata = w.parent / "metadata.yaml"

## 修改cfg/default.yaml文件

83行

    format: rknn #torchscript # (str) target format, e.g. torchscript|onnx|openvino|engine|coreml|saved_model|pb|tflite|edgetpu|tfjs|paddle|mnn|ncnn|imx|rknn|executorch

## 修改ultralytics/nn/modules/head.py文件

141-163行

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.export and self.format == 'rknn':
            if self.end2end:
                y = []
                for i in range(self.nl):
                    y.append(self.one2one_cv2[i](x[i]))
                    cls = torch.sigmoid(self.one2one_cv3[i](x[i]))
                    cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
                    y.append(cls)
                    y.append(cls_sum)
                return y
            else:
                y = []
                for i in range(self.nl):
                    y.append(self.cv2[i](x[i]))
                    cls = torch.sigmoid(self.cv3[i](x[i]))
                    cls_sum = torch.clamp(cls.sum(1, keepdim=True), 0, 1)
                    y.append(cls)
                    y.append(cls_sum)
                return y

## 修改ultralytics-8.4.3\ultralytics\engine\exporter.py文件
    
1265行export_rknn函数

    @try_export
    def export_rknn(self, prefix=colorstr("RKNN:")):
        """Export YOLO model to RKNN format."""
        LOGGER.info(f'\n{prefix} starting export with torch {torch.__version__}...')

        # ts = torch.jit.trace(self.model, self.im, strict=False)
        # f = str(self.file).replace(self.file.suffix, f'_rknnopt.torchscript')
        # torch.jit.save(ts, str(f))

        f = str(self.file).replace(self.file.suffix, f'.onnx')
        opset_version = self.args.opset # or get_latest_opset()
        torch.onnx.export(
            self.model,
            self.im[0:1,:,:,:],
            f,
            verbose=False,
            opset_version=14,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'])

        LOGGER.info(f'\n{prefix} feed {f} to RKNN-Toolkit or RKNN-Toolkit2 to generate RKNN model.\n' 
                    'Refer https://github.com/airockchip/rknn_model_zoo/tree/main/examples/')
        return f, None