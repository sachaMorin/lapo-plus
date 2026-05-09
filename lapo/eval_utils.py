import wandb
import torch
import torch.nn.functional as F

from env_utils import procgen_action_meanings


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy (in nats) from logits in a numerically stable way.

    Args:
        logits: Tensor of shape (..., num_classes)

    Returns:
        Tensor of shape (...) with the entropy for each sample.
    """
    # Compute log-probabilities in a stable way
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def test_step_policy(step, policy, test_iter, logger, idm=None):
    with torch.inference_mode():
        policy.eval()

        test_batch = next(test_iter)

        pred_test_la = policy(
            test_batch["obs"][:, -2]
        )  # the -2 selects last the pre-transition ob
        pred_test_ta_logits = policy.decoder(pred_test_la)

        test_ta = test_batch["ta"][:, -2]
        test_loss_expert = F.cross_entropy(pred_test_ta_logits, test_ta)

        pred_test_ta = pred_test_ta_logits.argmax(dim=-1)
        test_acc_expert = (pred_test_ta == test_ta).float().mean().item()

        test_entropy = entropy_from_logits(pred_test_ta_logits).mean().item()
        logger(
            step=step,
            test_loss_expert=test_loss_expert,
            test_acc_expert=test_acc_expert,
            test_entropy=test_entropy,
        )

        if idm is not None:
            idm.label(test_batch)
            idm_pred = test_batch["pred_ta"]
            test_loss_idm = F.cross_entropy(pred_test_ta_logits, idm_pred.argmax(-1))
            test_acc_idm = (pred_test_ta == idm_pred.argmax(-1)).float().mean().item()
            logger(step=step, test_loss_idm=test_loss_idm, test_acc_idm=test_acc_idm)


def test_step_idm(step, idm, test_iter, logger):
    with torch.inference_mode():
        idm.eval()
        test_batch = next(test_iter)
        idm.label(test_batch)
        pred_test_ta_logits = test_batch["pred_ta"]
        test_ta = test_batch["ta"][:, -2]
        test_loss_expert = F.cross_entropy(pred_test_ta_logits, test_ta)
        test_acc_expert = (
            (pred_test_ta_logits.argmax(-1) == test_ta).float().mean().item()
        )
        test_entropy = entropy_from_logits(pred_test_ta_logits).mean().item()

    logger(
        step=step,
        test_loss_expert=test_loss_expert,
        test_acc_expert=test_acc_expert,
        test_entropy=test_entropy,
    )


def log_confusion_matrix_policy(step, policy, test_iter, logger, idm=None):
    with torch.inference_mode():
        policy.eval()

        test_batch = next(test_iter)

        pred_test_la = policy(
            test_batch["obs"][:, -2]
        )  # the -2 selects last the pre-transition ob
        pred_test_ta_logits = policy.decoder(pred_test_la)
        pred_test_ta = pred_test_ta_logits.argmax(dim=-1)

        test_ta = test_batch["ta"][:, -2]

        confusion_matrix_expert = wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_ta.cpu().numpy(),
            preds=pred_test_ta.cpu().numpy(),
            class_names=procgen_action_meanings.tolist(),
            title="Confusion Matrix vs Expert Labels",
        )
        logger(confusion_matrix_expert=confusion_matrix_expert, step=step)

        # Optional IDM relabeling evaluation.
        # Compare policy predictions to IDM predictions
        if idm is not None:
            idm.label(test_batch)
            idm_pred = test_batch["pred_ta"]
            confusion_matrix_idm = wandb.plot.confusion_matrix(
                probs=None,
                y_true=idm_pred.argmax(-1).cpu().numpy(),
                preds=pred_test_ta.cpu().numpy(),
                class_names=procgen_action_meanings.tolist(),
                title="Confusion Matrix vs IDM Labels",
            )
            logger(confusion_matrix_idm=confusion_matrix_idm, step=step)


def log_confusion_matrix_idm(step, idm, test_iter, logger):
    with torch.inference_mode():
        idm.eval()
        test_batch = next(test_iter)
        idm.label(test_batch)
        pred_test_ta_logits = test_batch["pred_ta"]
        test_ta = test_batch["ta"][:, -2]
        confusion_matrix_expert = wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_ta.cpu().numpy(),
            preds=pred_test_ta_logits.argmax(-1).cpu().numpy(),
            class_names=procgen_action_meanings.tolist(),
            title="Confusion Matrix vs Expert Labels",
        )
    logger(confusion_matrix_expert=confusion_matrix_expert, step=step)
