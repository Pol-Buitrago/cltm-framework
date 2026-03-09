# utils_logging.py
import os
import logging
import numpy as np
import pandas as pd
import math
import torch
from accelerate import Accelerator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%y-%m-%d %H:%M:%S',
)

# Stream handler básico
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.DEBUG)
logger_stream_handler.setFormatter(logger_formatter)
logger.addHandler(logger_stream_handler)


class LoggerHelper:
    """
    Clase contenedora para manejar logging, dumping de log_history y gradientes.
    """
    def __init__(self, accelerator: Accelerator, ft_model_output_dir: str = None):
        self.accelerator = accelerator
        self.ft_model_output_dir = ft_model_output_dir

        self.logger_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%y-%m-%d %H:%M:%S',
        )

    def log_on_main(self, text, type="info"):
        if self.accelerator.is_local_main_process:
            if type == "debug":
                logger.debug(text)
            elif type == "warning":
                logger.warning(text)
            else:
                logger.info(text)

    def dump_log_history(self, trainer, main_metric="eval_wer"):
        """
        Compacta y correcta para volcar trainer.state.log_history a CSV.
        Agrupa por epoch, calcula media de train_loss y conserva eval_*.
        Corrección: prioriza trainer-provided 'train_loss' si existe,
        y sólo usa mean(loss_list) si no existe.
        """
        self.log_on_main("Dumping log_history...")

        if not self.accelerator.is_local_main_process:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    self.accelerator.wait_for_everyone()
            except Exception:
                pass
            return

        log_history = trainer.state.log_history
        # debug rápido: si quieres ver las últimas entradas uncomment:
        # self.log_on_main(f"DEBUG log_history tail: {log_history[-10:]}")

        buckets = {}

        # Agrupar por bucket de epoch (usar floor para agrupar pasos dentro de la misma epoch)
        for item in log_history:
            epoch_val = item.get("epoch", None)
            key = "epoch_unknown" if epoch_val is None else f"epoch_{max(1, int(math.floor(float(epoch_val))))}"
            if key not in buckets:
                buckets[key] = {"_loss_list": [], "_last": {}}

            # Guardamos 'loss' por separado y todo lo demás en _last
            for k, v in item.items():
                if k == "loss":
                    try:
                        buckets[key]["_loss_list"].append(float(v))
                    except Exception:
                        pass
                else:
                    # siempre actualizamos _last con la última aparición de esa clave
                    buckets[key]["_last"][k] = v

        # Construir rows finales
        rows = []
        # ordenar keys por número de epoch (epoch_1, epoch_2, ...)
        def _epoch_sort_key(idx_name):
            try:
                parts = idx_name.split("_")
                return float(parts[1]) if len(parts) > 1 and parts[1].replace('.', '', 1).isdigit() else 0.0
            except Exception:
                return 0.0

        for key in sorted(buckets.keys(), key=_epoch_sort_key):
            data = {}
            last = buckets[key]["_last"].copy()  # copia para no mutar original

            # incorporar todas las claves de last
            for k, v in last.items():
                data[k] = v

            # compute train_loss: prioridad a last.get('train_loss') si existe,
            # si no, usar la media de _loss_list; si nada, NaN
            loss_list = buckets[key].get("_loss_list", [])
            if "train_loss" in last and last["train_loss"] is not None:
                try:
                    data["train_loss"] = float(last["train_loss"])
                except Exception:
                    data["train_loss"] = float("nan")
            elif loss_list:
                data["train_loss"] = float(np.mean(loss_list))
            else:
                data["train_loss"] = float("nan")

            data["train_loss_count"] = len(loss_list) if loss_list else 0

            rows.append((key, data))

        if not rows:
            self.log_on_main("No log_history entries found.")
            return

        df_log_history = pd.DataFrame.from_dict({k: v for k, v in rows}, orient="index")

        # Ordenar por número de epoch si procede
        try:
            if "epoch" in df_log_history.columns:
                df_log_history.sort_values("epoch", inplace=True)
            else:
                df_log_history["_epoch_sort"] = [_epoch_sort_key(ix) for ix in df_log_history.index]
                df_log_history.sort_values("_epoch_sort", inplace=True)
                df_log_history.drop(columns=["_epoch_sort"], inplace=True)
        except Exception:
            pass

        # Guardar CSV
        if self.ft_model_output_dir:
            os.makedirs(self.ft_model_output_dir, exist_ok=True)
            out_csv = os.path.join(self.ft_model_output_dir, "log_history.csv")
            df_log_history.to_csv(out_csv, sep="\t", index=True)
            self.log_on_main(f"Saved log_history to {out_csv}")

        # Preview controlado
        self.log_on_main("-" * 50)
        self.log_on_main("df_log_history preview:")
        self.log_on_main(f"\n{df_log_history.head(30)}")
        self.log_on_main("-" * 50)

        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.accelerator.wait_for_everyone()
        except Exception:
            pass

        self.log_on_main("Dumping log_history done!")


    def log_grad_params(self, model):
        self.log_on_main("Computing gradient statistics...")

        named_parameters = model.named_parameters()
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in named_parameters:
            self.log_on_main(f"n={n}, p.requires_grad={p.requires_grad}, p.grad={p.grad}")
            if p.requires_grad and ("bias" not in n) and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())

        self.log_on_main(f"Done!")

        for layer, ave_grad, max_grad in zip(layers, ave_grads, max_grads):
            self.log_on_main(f"Layer {layer} gradients statistics: max={max_grad}, avg={ave_grad}")

        self.log_on_main("Computing gradient statistics done!")
