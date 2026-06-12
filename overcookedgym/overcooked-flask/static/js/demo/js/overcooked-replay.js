import * as Overcooked from "overcooked"
let OvercookedGame = Overcooked.OvercookedGame.OvercookedGame;
let OvercookedMDP = Overcooked.OvercookedMDP;
let Direction = OvercookedMDP.Direction;
let Action = OvercookedMDP.Action;
let PlayerState = OvercookedMDP.PlayerState; 
let OvercookedState = OvercookedMDP.OvercookedState; 
let ObjectState = OvercookedMDP.ObjectState;

let [NORTH, SOUTH, EAST, WEST] = Direction.CARDINAL;
let [STAY, INTERACT] = [Direction.STAY, Action.INTERACT];

let lookupActions = OvercookedMDP.lookupActions; 
let dictToState = OvercookedMDP.dictToState;

export default class OvercookedTrajectoryReplay{
    constructor ({
        container_id,
        trajectory,
        start_grid = [
            'XXXXXPXX',
            'O     2O',
            'T1     T',
            'XXXDPSXX'
        ],
        MAX_TIME = 1, //seconds
        cook_time=5,
        init_orders=null,
        completion_callback = () => {console.log("Time up")},
        timestep_callback = (data) => {},
        DELIVERY_REWARD = 20
    }) 
    {

    	let player_colors = {};
    	player_colors[0] = 'green';
        player_colors[1] = 'blue';

        this.game = new OvercookedGame({
            start_grid,
            container_id,
            assets_loc: "static/assets/",
            ANIMATION_DURATION: 200*.9,
            tileSize: 80,
            COOK_TIME: cook_time,
            explosion_time: Number.MAX_SAFE_INTEGER,
            DELIVERY_REWARD: DELIVERY_REWARD,
            player_colors: player_colors
        });
        this.init_orders = init_orders;
        console.log("Trajectory replay");
        this.observations = trajectory.ep_states[0];
        this.actions = trajectory.ep_actions[0];
        this.MAX_TIME = MAX_TIME;
        this.time_left = MAX_TIME;
        this.cur_gameloop = 0;
        this.score = 0;
        this.completion_callback = completion_callback;
        this.timestep_callback = timestep_callback;
        this.total_timesteps = this.observations.length - 1;
        this.is_playing = true;
        this.is_scrubbing = false;
        this.resume_after_scrub = false;
        this.last_step_time = Date.now();
        this.seconds_per_step = 0.5;
    }


    init() {
        this.game.init();
        this.activate_response_listener();
        this.render_step(0);
        this.set_playing(true);
        this.gameloop = setInterval(() => this.tick(), 50);
    }

    tick() {
        if (!this.is_playing || this.is_scrubbing) {
            return;
        }
        let now = Date.now();
        if ((now - this.last_step_time) / 1000 < this.seconds_per_step) {
            return;
        }
        if (this.cur_gameloop >= this.total_timesteps) {
            this.set_playing(false);
            return;
        }
        this.last_step_time = now;
        this.render_step(this.cur_gameloop + 1);
    }

    render_step(step) {
        this.cur_gameloop = Math.max(
            0, Math.min(this.total_timesteps, Number(step))
        );
        this.state = dictToState(this.observations[this.cur_gameloop]);
        if (this.state.order_list == null) {
            this.state.order_list = [];
        }
        this.game.drawState(this.state);

        this.time_left = this.total_timesteps - this.cur_gameloop;
        this.game.drawTimeLeft(this.time_left);
        document.getElementById("stepSlider").value = this.cur_gameloop;
        document.getElementById("stepLabel").textContent =
            `${this.cur_gameloop} / ${this.total_timesteps}`;
    }

    set_playing(is_playing) {
        this.is_playing = Boolean(is_playing);
        this.last_step_time = Date.now();
        let button = document.getElementById("playPause");
        if (button !== null) {
            button.textContent = this.is_playing ? "Pause" : "Play";
        }
    }

    toggle_playing() {
        this.set_playing(!this.is_playing);
    }

    restart() {
        this.render_step(0);
        this.set_playing(true);
    }

    close () {
        if (typeof(this.gameloop) !== 'undefined') {
            clearInterval(this.gameloop);
        }
        this.game.close();
        this.disable_response_listener();
        this.completion_callback();
    }

    activate_response_listener () {
        let slider = document.getElementById("stepSlider");
        slider.min = 0;
        slider.max = this.total_timesteps;
        slider.step = 1;
        slider.value = 0;

        $(slider).off(".overcookedReplay");
        $(slider).on(
            "mousedown.overcookedReplay touchstart.overcookedReplay",
            () => {
                this.is_scrubbing = true;
                this.resume_after_scrub = this.is_playing;
                this.set_playing(false);
            }
        );
        $(slider).on("input.overcookedReplay", (e) => {
            this.render_step(Number(e.target.value));
        });
        $(slider).on(
            "change.overcookedReplay mouseup.overcookedReplay touchend.overcookedReplay",
            () => {
                this.is_scrubbing = false;
                this.set_playing(this.resume_after_scrub);
            }
        );

        $(document).off(".overcookedReplay");
        $(document).on("keydown.overcookedReplay", (e) => {
            switch(e.which) {
            case 37: // left
                this.set_playing(false);
                this.render_step(this.cur_gameloop - 1);
                break;

            case 39: // right
                this.set_playing(false);
                this.render_step(this.cur_gameloop + 1);
                break;

            case 32: //space
                this.toggle_playing();
                break;
            default: return; // exit this handler for other keys
            }
            e.preventDefault(); // prevent the default action (scroll / move caret)
        });
    }

    disable_response_listener () {
        $(document).off(".overcookedReplay");
        $("#stepSlider").off(".overcookedReplay");
    }
}
