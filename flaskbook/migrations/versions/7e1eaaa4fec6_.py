"""empty message

Revision ID: 7e1eaaa4fec6
Revises: 1089635e8caa
Create Date: 2023-09-19 10:50:40.996655

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7e1eaaa4fec6'
down_revision = '1089635e8caa'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user_image_tags',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_image_id', sa.String(), nullable=True),
    sa.Column('tag_name', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_image_id'], ['user_images.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user_image_tags')
    # ### end Alembic commands ###
