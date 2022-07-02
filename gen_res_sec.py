#!/bin/env python

if __name__ == "__main__":
    prefix = "illustration"
    for train_loop in ["simple", "cycle", "harmonic"]:
        print(f"\n\n## {train_loop.capitalize()}")
        for model in ["pix", "dense"]:
            print(f"\n### {model.capitalize()}")
            for pret, pretrain in enumerate(["pretrain", "nopretrain"]):
                header = "Pretrain" if pret == 0 else "No pretrain"
                print(f"#### {header}")
                for m, mode in enumerate(["train", "eval"]):
                    dropout = "50% dropout" if m == 0 else "No dropout"
                    print(f"##### {dropout}")
                    print("<table><tbody>")
                    print("<tr>")
                    for file in ["mike_wazowski", "horny", "busya", "ll", "triplechad"]:
                        print('<td><img src="' + '/'.join((prefix, train_loop, model, pretrain, mode, file, ".png")) + '" width="128px"/></td>')
                    print("</tr><tr>")
                    for file in ["sigma", "ginger", "ka_52", "pig", "floppa"]:
                        print('<td><img src="' + '/'.join((prefix, train_loop, model, pretrain, mode, file, ".png")) + '" width="128px"/></td>')
                    print("</tr>")
                    print("</tbody></table>")

