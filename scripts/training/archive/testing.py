import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import T5Tokenizer
import json

from data_loader.data_loading import MSRDataset
from models.encoder_prep.encoder import TextEncoder
from models.slot_pooling_prep.slot_pooling import SlotPooling
from models.uv_heads_prep.u_head import UHead
from models.uv_heads_prep.v_head import VHead
from models.decoder_prep.decoder_x import DecoderX
from models.decoder_prep.decoder_y import DecoderY

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def info_nce_loss(u, upos):
    u = F.normalize(u, dim = -1)
    upos = F.normalize(upos, dim = -1)
    logits = torch.matmul(u, upos.T)/0.1
    labels = torch.arange(u.size(0), device = u.device)
    loss = F.cross_entropy(logits, labels)
    return loss



def forward(encoder_model, slot_pool_model, u_head, v_head, decoder_x, decoder_y, lambda_u, lambda_x, lambda_y, batch, device):
    input_ids = batch["x_input_ids"].to(device)
    attention_mask = batch["x_attention"].to(device)
    xpos_input_ids = batch["xpos_input_ids"].to(device)
    xpos_attention_mask = batch["xpos_attention"].to(device)
    labels_x = batch["x_input_ids"].to(device)
    labels_y = batch["y_input_ids"].to(device)
    y_attention_mask = batch["y_attention"].to(device)

    H = encoder_model(
        input_ids = input_ids,
        attention_mask = attention_mask
    )
    Hpos = encoder_model(
        input_ids = xpos_input_ids,
        attention_mask = xpos_attention_mask
    )
    # print("shape of outputs after encoder", H.shape) # 2, 64, 512 

    outputs = slot_pool_model(H, attention_mask)
    pos_outputs = slot_pool_model(Hpos, xpos_attention_mask)
    # print("shape of outputs after slot pooling", outputs.shape) #2, 8, 512

    u = u_head(outputs)
    upos = u_head(pos_outputs)
    v0 = v_head(outputs)
    # print("shape of u", u.shape) # 2, 128
    # print("shape of v0", v0.shape)# 2, 8, 512

    B, L, _ = v0.shape
    slot_mask = torch.ones((B,L), device = device) #attention mask for v0

    
    loss_x, logits_x = decoder_x(v0, slot_mask, labels_x) 
    loss_y, logits_y = decoder_y(u, labels_y)
    loss_nce = info_nce_loss(u, upos) 

    total_loss = (
        lambda_u * loss_nce +
        lambda_x* loss_x + 
        lambda_y * loss_y

    )

    return total_loss, logits_x, logits_y

    # return 
    # return outputs

def main():
    print("device", device)
    train_dataset = MSRDataset("./dataset_prep/outputs/first_100_train.json")
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle = True)
    val_dataset = MSRDataset("./dataset_prep/outputs/first_100_test.json")
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle = True)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    encoder_model = TextEncoder()
    encoder_model.to(device)
    slot_pool_model = SlotPooling(hidden_dim = encoder_model.hidden_dim_size, num_slots = 8)
    slot_pool_model.to(device)
    u_head = UHead(hidden_dim = encoder_model.hidden_dim_size, output_dim = 128)
    v_head = VHead(hidden_dim = encoder_model.hidden_dim_size)
    u_head.to(device)
    v_head.to(device)
    decoder_x = DecoderX()
    decoder_y = DecoderY(hidden_dim = encoder_model.hidden_dim_size, u_dim = 128, num_slots = 8)
    decoder_x.to(device)
    decoder_y.to(device)

    # batch = next(iter(dataloader))

    # forward(encoder_model, slot_pool_model, u_head, v_head, decoder_x, decoder_y, 1.0, 1.0, 1.0, batch, device)

    params = (
        list(encoder_model.parameters())+
        list(slot_pool_model.parameters())+
        list(u_head.parameters())+
        list(v_head.parameters())+
        list(decoder_x.parameters())+
        list(decoder_y.parameters())
    )
    optimizer = torch.optim.Adam(params, lr = 1e-4)
    epochs = 500

    for epoch in range(epochs):
        encoder_model.train() #add others - lazy

        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            loss, _, _ = forward(encoder_model, slot_pool_model, u_head, v_head, decoder_x, decoder_y, 1.0, 1.0, 1.0, batch, device)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss/len(train_loader)

        encoder_model.eval()
        total_val_loss = 0

        sample_outputs = None
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss, logits_x, logits_y = forward(encoder_model, slot_pool_model, u_head, v_head, decoder_x, decoder_y, 1.0, 1.0, 1.0, batch, device)
                total_val_loss += loss.item()
                if i==0:
                    sample_outputs = (batch, logits_x, logits_y)
        avg_val_loss = total_val_loss/len(val_loader)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print("------------")

        if sample_outputs is not None:
            batch, logits_x, logits_y = sample_outputs

            pred_ids_x = torch.argmax(logits_x, dim=-1)
            pred_ids_y = torch.argmax(logits_y, dim=-1)
            decoded_x = tokenizer.batch_decode(pred_ids_x, skip_special_tokens=True)
            decoded_y = tokenizer.batch_decode(pred_ids_y, skip_special_tokens=True)

            original_x = tokenizer.batch_decode(batch["x_input_ids"], skip_special_tokens=True)
            original_y = tokenizer.batch_decode(batch["y_input_ids"], skip_special_tokens=True)

            results = []
            for i in range(len(decoded_x)):
                results.append({
                    "original_x": original_x[i],
                    "v0": decoded_x[i],
                    "original_y": original_y[i],
                    "u": decoded_y[i]
                })

            with open(f"./final_outputs/epoch_{epoch+1}_samples.json", "w") as f:
                json.dump(results, f, indent=4)




    print("tested")

if __name__=="__main__":
    main()


