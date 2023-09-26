# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Predicate learning framework
#  Copyright (C) 2023. ETH ASL
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import tkinter as tk
from tkinter import ttk


class DummyPublisher:
    def publish(self, msg=None):
        raise NotImplementedError


class Application:
    def __init__(self, ManipulationCmd_MsgType, Randomization_MsgType=None):
        self.root = tk.Tk()
        self.feet = None
        self.meters = None
        self.running = tk.BooleanVar()
        self.pred_name = tk.StringVar()
        self.pred_args = tk.StringVar()
        self.moving_object_name = tk.StringVar()

        self.ManipulationCmd_MsgType = ManipulationCmd_MsgType
        self.mani_pub = DummyPublisher()
        self.logger = None

        self.confirm_button = None
        self.deny_button = None
        self.progress_bar = None

        self.Randomization_MsgType = Randomization_MsgType
        self.randomization_data = None
        self.randomize_args = tk.StringVar()
        self.randomize_ignore_objects = tk.StringVar()
        self.rand_pub = DummyPublisher()

        self.generator_model = tk.StringVar()
        self.gen_model_dropdown = None

        self.demo_id = tk.StringVar()

        self.info_pub = DummyPublisher()

        self.setup_window()

    def setup_window(self):
        self.root.title("Simulation Control")

        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Button(mainframe, text="Run", command=self._run_sim).grid(row=1, column=1)
        ttk.Button(mainframe, text="Stop", command=self._stop_sim).grid(row=1, column=2)
        ttk.Label(mainframe, text="Simulation status:").grid(row=1, column=3)
        ttk.Label(mainframe, textvariable=self.running).grid(row=1, column=4)
        ttk.Label(mainframe, text="Predicate name:").grid(column=1, row=2, sticky=tk.W)
        ttk.Entry(mainframe, width=14, textvariable=self.pred_name).grid(
            column=2, row=2, sticky=tk.W
        )
        ttk.Label(mainframe, text="Predicate arguments:").grid(column=3, row=2)
        ttk.Entry(mainframe, width=14, textvariable=self.pred_args).grid(
            column=4, row=2, sticky=tk.W
        )

        # Demonstration interface
        demo_frame = ttk.Frame(mainframe, borderwidth=5, relief="ridge")
        demo_frame.grid(row=6, column=1, columnspan=2)
        ttk.Label(demo_frame, text="Demonstration Interface").grid(
            column=1, row=0, columnspan=3
        )
        ttk.Button(
            demo_frame, text="+ Demonstration", command=lambda: self._snapshot(0, True)
        ).grid(column=1, row=5)
        ttk.Button(
            demo_frame, text="- Demonstration", command=lambda: self._snapshot(0, False)
        ).grid(column=2, row=5)
        ttk.Button(
            demo_frame, text="Extract features", command=lambda: self._snapshot(1)
        ).grid(column=1, row=6)
        ttk.Button(demo_frame, text="Print info", command=self._print_info).grid(
            column=2, row=6
        )
        # ttk.Button(demo_frame, text="Classify", command=lambda: self._snapshot(3)).grid(
        #     column=3, row=6
        # )
        self.progress_bar = ttk.Progressbar(
            demo_frame, orient=tk.HORIZONTAL, length=200, mode="indeterminate"
        )
        self.progress_bar.grid(row=8, column=1, columnspan=3)
        for child in demo_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Scene interface
        scene_frame = ttk.Frame(mainframe, borderwidth=5, relief="ridge")
        scene_frame.grid(row=7, column=1, columnspan=2)
        ttk.Label(scene_frame, text="Scene Interface").grid(
            row=0, column=0, columnspan=3
        )
        ttk.Button(
            scene_frame, text="Reset scene", command=lambda: self._snapshot(6)
        ).grid(row=1, column=0, columnspan=3)
        ttk.Entry(scene_frame, width=14, textvariable=self.demo_id).grid(
            row=2, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Button(scene_frame, text="Restore demo", command=self._restore_scene).grid(
            row=2, column=2, columnspan=1
        )
        ttk.Button(
            scene_frame, text="Restore next demo", command=lambda: self._snapshot(8)
        ).grid(row=3, column=0, columnspan=1)
        ttk.Button(
            scene_frame, text="Restore features", command=lambda: self._snapshot(9)
        ).grid(row=3, column=2, columnspan=1)
        for child in scene_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Manipulation interface
        manipulation_frame = ttk.Frame(mainframe, borderwidth=5, relief="ridge")
        manipulation_frame.grid(row=5, column=1, columnspan=2)

        ttk.Label(manipulation_frame, text="Manual Manipulation Interface").grid(
            column=1, row=1, columnspan=2
        )
        ttk.Label(manipulation_frame, text="Target name:").grid(
            column=1, row=2, sticky=tk.W
        )
        ttk.Entry(
            manipulation_frame, width=10, textvariable=self.moving_object_name
        ).grid(column=2, row=2, sticky=tk.W)
        ttk.Label(manipulation_frame, text="Ctrl-Alt-<key> to translate").grid(
            column=1, row=3, columnspan=2
        )
        ttk.Label(manipulation_frame, text="Ctrl-Alt-Shift-<key> to rotate").grid(
            column=1, row=4, columnspan=2
        )
        for child in manipulation_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Generator interface
        generator_frame = ttk.Frame(mainframe, borderwidth=5, relief="ridge")
        generator_frame.grid(row=5, column=3, columnspan=2)
        ttk.Label(generator_frame, text="Sample Generation Interface").grid(
            column=1, row=1, columnspan=2
        )
        ttk.Label(generator_frame, text="Model selection:").grid(
            column=1, row=2, sticky=tk.W
        )
        self.gen_model_dropdown = ttk.Combobox(
            generator_frame, textvariable=self.generator_model
        )
        self.gen_model_dropdown.grid(row=2, column=2, sticky=tk.W)
        self.gen_model_dropdown.state(["readonly"])
        ttk.Button(
            generator_frame,
            text="Refresh",
            command=lambda: self._get_generator_models_gui("on_supporting_ig"),
        ).grid(row=3, column=1, columnspan=1)
        ttk.Button(
            generator_frame,
            text="Set parameters",
            command=self._set_generator_model_gui,
        ).grid(row=3, column=2, columnspan=1)
        ttk.Button(
            generator_frame, text="Sample noise", command=lambda: self._snapshot(5)
        ).grid(row=4, column=1, columnspan=1)
        ttk.Button(
            generator_frame, text="Generate", command=lambda: self._snapshot(4)
        ).grid(row=4, column=2)
        # ttk.Entry(
        #     manipulation_frame, width=10, textvariable=self.moving_object_name
        # ).grid(column=2, row=2, sticky=tk.W)
        # ttk.Label(manipulation_frame, text="Ctrl-Alt-<key> to translate").grid(
        #     column=1, row=3, columnspan=2
        # )
        # ttk.Label(manipulation_frame, text="Ctrl-Alt-Shift-<key> to rotate").grid(
        #     column=1, row=4, columnspan=2
        # )
        for child in generator_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Randomize frame
        randomize_frame = ttk.Frame(mainframe, borderwidth=5, relief="ridge")
        randomize_frame.grid(row=6, column=3, rowspan=2, columnspan=2)
        ttk.Label(randomize_frame, text="Randomization Interface").grid(
            column=1, row=1, columnspan=10
        )
        col_label = ["Pos.", "Ori.", "Joint"]
        col_magnitude = ["0", "S", "L"]
        for i, l in enumerate(col_label):
            col = 2 + i * 3
            ttk.Label(randomize_frame, text=l).grid(row=2, column=col, columnspan=3)
            for j, lj in enumerate(col_magnitude):
                col = 2 + j + 3 * i
                ttk.Label(randomize_frame, text=lj).grid(row=3, column=col)
        row_label = ["Arg0", "Arg1", "Others"]
        for i, r in enumerate(row_label):
            ttk.Label(randomize_frame, text=r).grid(row=4 + i, column=1, sticky=tk.W)
        self.randomization_data = [
            [tk.IntVar() for _ in range(len(col_label))] for _ in range(len(row_label))
        ]
        for i in range(len(row_label)):
            for j in range(len(col_label)):
                self.randomization_data[i][j].set(0)
                for k in range(len(col_magnitude)):
                    ttk.Radiobutton(
                        randomize_frame, variable=self.randomization_data[i][j], value=k
                    ).grid(row=4 + i, column=2 + k + 3 * j)
        ttk.Label(randomize_frame, text="Arguments:").grid(
            column=1, row=7, columnspan=4, sticky=tk.W
        )
        ttk.Entry(randomize_frame, width=12, textvariable=self.randomize_args).grid(
            column=4, row=7, columnspan=5, sticky=tk.W
        )
        ttk.Label(randomize_frame, text="Ignore:").grid(
            column=1, row=8, columnspan=4, sticky=tk.W
        )
        ttk.Entry(
            randomize_frame, width=12, textvariable=self.randomize_ignore_objects
        ).grid(column=4, row=8, columnspan=5, sticky=tk.W)
        ttk.Button(
            randomize_frame, text="Randomize objects", command=self._randomize
        ).grid(row=9, column=1, columnspan=10)
        for child in randomize_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # -----------------------------------------------------

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Key bindings
        self.root.bind("<Return>", self._run_sim)
        self._bind_key("<Control-Alt-KeyPress-u>", (0, "trans", "fast", -1))
        self._bind_key("<Control-Alt-KeyPress-i>", (0, "trans", "slow", -1))
        self._bind_key("<Control-Alt-KeyPress-o>", (0, "trans", "slow", 1))
        self._bind_key("<Control-Alt-KeyPress-p>", (0, "trans", "fast", 1))
        self._bind_key("<Control-Alt-KeyPress-j>", (1, "trans", "fast", -1))
        self._bind_key("<Control-Alt-KeyPress-k>", (1, "trans", "slow", -1))
        self._bind_key("<Control-Alt-KeyPress-l>", (1, "trans", "slow", 1))
        self._bind_key("<Control-Alt-KeyPress-semicolon>", (1, "trans", "fast", 1))
        self._bind_key("<Control-Alt-KeyPress-m>", (2, "trans", "fast", -1))
        self._bind_key("<Control-Alt-KeyPress-comma>", (2, "trans", "slow", -1))
        self._bind_key("<Control-Alt-KeyPress-period>", (2, "trans", "slow", 1))
        self._bind_key("<Control-Alt-KeyPress-slash>", (2, "trans", "fast", 1))
        self._bind_key("<Control-Alt-Shift-KeyPress-U>", (0, "rot", "fast", -1))
        self._bind_key("<Control-Alt-Shift-KeyPress-I>", (0, "rot", "slow", -1))
        self._bind_key("<Control-Alt-Shift-KeyPress-O>", (0, "rot", "slow", 1))
        self._bind_key("<Control-Alt-Shift-KeyPress-P>", (0, "rot", "fast", 1))
        self._bind_key("<Control-Alt-Shift-KeyPress-J>", (1, "rot", "fast", -1))
        self._bind_key("<Control-Alt-Shift-KeyPress-K>", (1, "rot", "slow", -1))
        self._bind_key("<Control-Alt-Shift-KeyPress-L>", (1, "rot", "slow", 1))
        self._bind_key("<Control-Alt-Shift-KeyPress-colon>", (1, "rot", "fast", 1))
        self._bind_key("<Control-Alt-Shift-KeyPress-M>", (2, "rot", "fast", -1))
        self._bind_key("<Control-Alt-Shift-KeyPress-less>", (2, "rot", "slow", -1))
        self._bind_key("<Control-Alt-Shift-KeyPress-greater>", (2, "rot", "slow", 1))
        self._bind_key("<Control-Alt-Shift-KeyPress-question>", (2, "rot", "fast", 1))

    def _bind_key(self, key, params):
        self.root.bind(key, lambda e: self._move_object(*params))

    def _call_setbool_service(self, data):
        raise NotImplementedError

    def _call_hlpguicommand_service(
        self, command, pred_name, pred_args, label, other_data
    ):
        raise NotImplementedError

    def _get_generator_models_gui(self, pred_name):
        names = self._get_generator_models(pred_name)
        self.gen_model_dropdown["values"] = names

    def _get_generator_models(self, pred_name):
        raise NotImplementedError

    def _set_generator_model_gui(self):
        self._set_generator_model(self.generator_model.get())

    def _set_generator_model(self, mdl_name):
        raise NotImplementedError

    def _run_sim(self, *args):
        self.logger.info("Sent run command")
        res = self._call_setbool_service(True)
        if not res.success:
            raise RuntimeError
        self.running.set(True)
        self.root.bind("<Return>", self._stop_sim)

    def _stop_sim(self, *args):
        self.logger.info("Sent stop command")
        res = self._call_setbool_service(False)
        if res.success:
            raise RuntimeError
        self.running.set(False)
        self.root.bind("<Return>", self._run_sim)

    def _snapshot(self, cmd, label=True, other_data=""):
        self.progress_bar.start()
        self._call_hlpguicommand_service(
            cmd, self.pred_name.get(), self.pred_args.get(), label, other_data
        )
        self.progress_bar.stop()

    def _restore_scene(self):
        self._snapshot(7, other_data=self.demo_id.get())

    def _move_object(self, axis, mode, speed, direction):
        translation = [0.0] * 3
        rotation = [0.0] * 3
        if mode == "trans":
            if speed == "slow":
                translation[axis] = 0.02 * direction
            else:
                translation[axis] = 0.2 * direction
        else:
            if speed == "slow":
                rotation[axis] = 1.0 * direction
            else:
                rotation[axis] = 5.0 * direction

        msg = self.ManipulationCmd_MsgType()
        msg.target_name = self.moving_object_name.get()
        msg.translation = translation
        msg.rotation = rotation
        self.mani_pub.publish(msg)

    def _randomize(self):
        mat = [el.get() for row in self.randomization_data for el in row]
        msg = self.Randomization_MsgType()
        msg.arguments = self.randomize_args.get()
        msg.selectors = mat
        msg.ignore_objects = self.randomize_ignore_objects.get()
        self.rand_pub.publish(msg)

    def _print_info(self):
        self.info_pub.publish()

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = Application(None)
    app.mainloop()
